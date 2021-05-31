#  from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

from losses import get_lossfn
from optimizers import get_optimizer, get_scheduler
from tokenization_kobert import KoBertTokenizer


class BertBase(nn.Module):
    def __init__(self, args, loss_fn):
        super().__init__()
        self.backbone = None
        self.args = args
        self.loss_fn = loss_fn

    def make_inputs(self, batch):
        inputs = {
            "input_ids": batch["input_ids"].to(self.args.device),
            "attention_mask": batch["attention_mask"].to(self.args.device),
        }

        if self.args.ms_name not in ["distilkobert", "xlmroberta"]:
            inputs["token_type_ids"] = batch["token_type_ids"].to(self.args.device)

        return inputs

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def one_step_inference(self, batch):
        inputs = self.make_inputs(batch)
        preds = self.backbone(**inputs, return_dict=True)
        return preds

    def train(self, dataloader):
        self.backbone.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            self.optimizer.zero_grad()

            labels = batch["labels"].to(self.args.device)

            preds = self.one_step_inference(batch)

            loss = self.loss_fn(preds.logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.args.clip)

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader, return_keys=["loss", "acc"]):
        self.backbone.eval()
        epoch_loss = 0.0

        results = dict()
        all_logits = torch.tensor([]).to(self.args.device)
        all_labels, all_preds = [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                labels = batch["labels"].to(self.args.device)
                preds = self.one_step_inference(batch)

                if "loss" in return_keys:
                    loss = self.loss_fn(preds.logits, labels)
                    epoch_loss += loss.item()

                all_logits = torch.cat((all_logits, preds.logits.detach()))
                all_labels.extend(labels.detach().cpu().tolist())

        all_preds = all_logits.detach().argmax(-1).cpu().tolist()

        assert len(all_labels) == len(all_preds)

        if "loss" in return_keys:
            results["loss"] = epoch_loss / len(dataloader)

        if "acc" in return_keys:
            results["acc"] = accuracy_score(all_labels, all_preds)

        if "preds" in return_keys:
            results["preds"] = all_preds

        if "logits" in return_keys:
            results["logits"] = all_logits.detach().cpu().numpy()

        return results


class KoELectraClassifier(BertBase):
    def __init__(self, args, loss_fn):
        super().__init__(args, loss_fn)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = 42
        self.backbone = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)


def load_model_and_tokenizer(args):

    loss_fn = get_lossfn(args)

    # 에라 모르겠다.
    model = KoELectraClassifier(args, loss_fn)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    model.set_optimizer(optimizer)
    model.set_scheduler(scheduler)

    model.to(args.device)

    if args.model_name_or_path in ["monologg/distilkobert", "monologg/kobert"]:
        vocab_file = os.path.join("./model_vocab", args.ms_name + ".model")
        vocab_txt = os.path.join("./model_vocab", args.ms_name + ".txt")
        tokenizer = KoBertTokenizer(vocab_file, vocab_txt)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer
