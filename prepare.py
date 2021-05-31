import pickle
import os.path as p
from typing import Callable

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Sampler


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count: dict = {}

        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def _get_label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)


def pick_one_dataset(args, is_train=True):
    data_path = p.join(args.data_dir, args.data_kind)

    if is_train:
        idx = args.dataset_idx

        train_path = p.join(data_path, f"train-{idx}.pkl")
        valid_path = p.join(data_path, f"valid-{idx}.pkl")

        with open(train_path, "rb") as f:
            train_dataset = pickle.load(f)

        with open(valid_path, "rb") as f:
            valid_dataset = pickle.load(f)

        return train_dataset, valid_dataset
    else:
        test_path = p.join(data_path, "test.pkl")

        with open(test_path, "rb") as f:
            test_dataset = pickle.load(f)

        return test_dataset


def load_test_dataloader(args, tokenizer):
    test_dataset = pick_one_dataset(args, is_train=False)
    tt_dataset = tokenized_dataset(args, test_dataset, tokenizer)
    tt_dataset = RE_Dataset(tt_dataset, test_dataset["labels"])

    test_dataloader = DataLoader(
        tt_dataset, batch_size=32, pin_memory=True, num_workers=1, shuffle=False
    )

    return test_dataloader


def load_sample(args, tokenizer, is_train=True):

    if is_train:
        _, dataset = pick_one_dataset(args, is_train=is_train)
    else:
        dataset = pick_one_dataset(args, is_train=is_train)

    tv_dataset = tokenized_dataset(args, dataset, tokenizer)
    re_tv_dataset = RE_Dataset(tv_dataset, dataset["labels"])

    batch = re_tv_dataset[0]

    inputs = {
        "input_ids": batch["input_ids"].to(args.device).unsqueeze(0),
        "attention_mask": batch["attention_mask"].to(args.device).unsqueeze(0),
    }

    if args.ms_name not in ["distilkobert", "xlmroberta"]:
        inputs["token_type_ids"] = batch["token_type_ids"].to(args.device).unsqueeze(0)

    labels = batch["labels"].to(args.device).unsqueeze(0)

    return inputs, labels


def load_dataloader(args, tokenizer):
    train_dataset, valid_dataset = pick_one_dataset(args, is_train=True)

    tt_dataset = tokenized_dataset(args, train_dataset, tokenizer)
    tv_dataset = tokenized_dataset(args, valid_dataset, tokenizer)

    re_tt_dataset = RE_Dataset(tt_dataset, train_dataset["labels"])
    re_tv_dataset = RE_Dataset(tv_dataset, valid_dataset["labels"])

    def callback_get_label(dataset, idx):
        return dataset._get_label(idx)

    if args.use_sampler is True:
        train_dataloader = DataLoader(
            re_tt_dataset,
            batch_size=args.batch_size,
            #  shuffle=True,
            pin_memory=True,  # Load Faster, If Use GPU
            num_workers=1,  # 이미 Memory에 다 올렸는데, 굳이 개수가 많을 필요가 있을까?
            sampler=ImbalancedDatasetSampler(
                re_tt_dataset, callback_get_label=callback_get_label
            ),
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            re_tt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,  # Load Faster, If Use GPU
            num_workers=1,  # 이미 Memory에 다 올렸는데, 굳이 개수가 많을 필요가 있을까?
            drop_last=True,
        )

    valid_dataloader = DataLoader(
        re_tv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    return train_dataloader, valid_dataloader


def tokenized_dataset(args, dataset, tokenizer):
    concat_entity = []

    if args.model_name_or_path not in ["monologg/distilkobert", "monologg/kobert"]:
        for idx, (e01, e02) in enumerate(zip(dataset["e1"], dataset["e2"])):
            #  concat_entity.append(e01 + "[SEP]" + e02)
            concat_entity.append(e01 + tokenizer.special_tokens_map["sep_token"] + e02)

        print(tokenizer.tokenize(dataset["words"][0]))

        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["words"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,  # 길수록 속도 느려짐
            add_special_tokens=True,
        )
    else:
        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []

        special_tokens_count = 1

        for idx, (e01, e02, words) in enumerate(
            zip(dataset["e1"], dataset["e2"], dataset["words"])
        ):
            e1_tokens = tokenizer.tokenize(e01)
            e2_tokens = tokenizer.tokenize(e02)
            word_tokens = tokenizer.tokenize(words)

            tokens = (
                ["[CLS]"] + e1_tokens + ["[SEP]"] + e2_tokens + ["[SEP]"] + word_tokens
            )

            if len(tokens) > args.max_seq_length - special_tokens_count:
                tokens = tokens[: (args.max_seq_length - special_tokens_count)]

            tokens += ["[SEP]"]

            token_type_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1] * len(input_ids)

            padding_length = args.max_seq_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length

            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length  # 질문 문제가 아니라서..

            assert len(input_ids) == args.max_seq_length
            assert len(attention_mask) == args.max_seq_length
            assert len(token_type_ids) == args.max_seq_length

            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append(attention_mask)

        tokenized_sentences = {
            "input_ids": torch.tensor(all_input_ids),
            "token_type_ids": torch.tensor(all_token_type_ids),
            "attention_mask": torch.tensor(all_attention_mask),
        }

    return tokenized_sentences


if __name__ == "__main__":
    from config import get_args
    from networks import load_model_and_tokenizer

    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)
    dataset, valid_dataset = pick_one_dataset(args, is_train=True)

    tv_dataset = tokenized_dataset(args, valid_dataset, tokenizer)

    #  for idx, (e01, e02, words, label) in enumerate(
    #      zip(dataset["e1"], dataset["e2"], dataset["words"], dataset["labels"])
    #  ):
    #      print(e01, e02, words, label)
    #      break
