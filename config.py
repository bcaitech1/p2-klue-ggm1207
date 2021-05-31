import os
import argparse
from functools import partial

import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="2-stage hypterparameter")
    pa = partial(parser.add_argument)

    # pathes

    if os.path.abspath(".").split("/")[1] == "opt":
        # server pathes
        pa("--weight_dir", default="/opt/ml/P-Stage/2-STAGE/weights")
        pa("--data_dir", default="/opt/ml/input/data")
        pa("--data_kind", default="dataset_v1")
        pa("--strategy", default="st00")
        pa("--submit_dir", default="/opt/ml/P-Stage/2-STAGE/submits")
    else:
        pa("--weight_dir", default="/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/weights")
        pa("--data_dir", default="/home/j-gunmo/storage/data/input/data/")
        pa("--data_kind", default="dataset_v1")
        pa("--strategy", default="st00")
        pa("--submit_dir", default="/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/submits")

    # ops hypterparameter: Uses when (not training)

    pa("--dataset_idx", default=0)
    pa("--num_labels", default=42, type=int)
    pa("--auto_sub", default=False, type=str2bool)
    pa("--debug", default=False, type=str2bool)

    # early stopping
    pa("--delta", default=0, type=int)
    pa("--patience", default=5, type=int)

    # train hypterparameters: Uses when training
    pa("--ms_name", default="koelectra")
    pa("--model_name_or_path", default="monologg/koelectra-base-v3-discriminator")
    pa("--use_sampler", default=True, type=str2bool)

    # loss, optimizer, scheduler
    pa("--loss_name", default="CE")
    pa("--optimizer", default="adamw")
    pa("--scheduler", default="warm_up")

    pa("--epochs", default=50, type=int)
    pa("--batch_size", default=64, type=int)
    pa("--warmup_steps", default=500, type=int)
    pa("--weight_decay", default=0.01, type=float)
    pa("--clip", default=1.0, type=float)
    pa("--max_seq_length", default=128, type=int)

    # batch sizes: 8, 16, 32, 64, 128
    # learning rates: 3e-4. 1e-4. 5e-5. 3e-5

    pa("--loss_hp", default={"reduction": "sum"})
    pa("--optimizer_hp", default={"lr": 5e-5, "eps": 1e-8}, type=dict)

    #  pa(
    #      "--scheduler_hp",
    #      default={
    #          "first_cycle_steps": 8,
    #          "cycle_mult": 1.0,
    #          "warmup_steps": 2,
    #          "gamma": 0.5,
    #      },
    #      type=dict,
    #  )

    pa("--scheduler_hp", default={"num_warmup_steps": 0}, type=dict)

    # parsing
    args, unknown = parser.parse_known_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
