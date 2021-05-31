import os
import glob
import random
import os.path as p
from argparse import Namespace

import torch
import numpy as np


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if use multi-GPU

    np.random.seed(SEED)
    random.seed(SEED)

    # 연산 처리 속도가 감소된다고 한다.
    #  torch.backends.cudnn.deterministic = True
    #  torch.backends.cudnn.benchmark = False


def custom_model_save(model, save_path):
    torch.save(model.state_dict(), save_path)


def get_auto_save_path(args):
    if isinstance(args, dict):
        args = Namespace(**args)

    prefix = "_".join([args.strategy, args.ms_name])
    prefix = p.join(args.weight_dir, prefix)

    s_cnt = len(glob.glob(prefix + "*"))
    save_path = f"{prefix}_{s_cnt:03}.pth"
    base_name = os.path.basename(save_path)[:-4]

    return save_path, base_name


def update_args(args, strategy, strat):
    """ this function use for hp search, Parallelism should be considered """

    if isinstance(args, Namespace):
        args = vars(args)

    args.update(strat[strategy])
    args["dataset_idx"] = random.randint(0, 4)  # 0 ~ 4

    save_path, base_name = get_auto_save_path(args)

    args["wandb"] = {"project": "p-stage-2", "api_key": "b9adc17bf9dff02b1aa29666268b7ab9ccaf2e56", "name": base_name}

    args["save_path"] = save_path
    args["base_name"] = base_name

    return args


class EarlyStopping:
    def __init__(self, args, verbose=True):
        self.args = args
        self.counter = 0
        self.verbose = verbose
        self.dalta = args.delta  # Delta값 기준으로 성능 개선을 판단한다.
        self.early_stop = False
        self.patience = args.patience
        self.best_valid_loss = float("inf")

        self.model_save_path = args.save_path

    def __call__(self, train_loss, valid_loss, valid_acc, model):

        if self.verbose:
            print(f"\tTrain Loss: {train_loss:.4f}")
            print(f"\tValidation Loss: {valid_loss:.4f}")
            print(f"\tValidation Acc: {valid_acc:.4f}")

        if valid_loss < self.best_valid_loss - self.dalta:
            self.save_checkpoint(valid_loss, model)
            self.best_valid_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, valid_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_valid_loss:.4f} --> {valid_loss:.4f}).  Saving model ...")

        torch.save(model.state_dict(), self.args.save_path)
