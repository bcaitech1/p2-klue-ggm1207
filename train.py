import time
import traceback
from argparse import Namespace

import wandb

import hp_space
from config import get_args
from database import execute_query
from slack import hook_fail_strategy
from networks import load_model_and_tokenizer
from inference import if_best_score_auto_submit
from prepare import load_sample, load_dataloader
from utils import EarlyStopping, get_auto_save_path, update_args


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run(args, model, train_dataloader, valid_dataloader):
    """ train, evaluate for range(epochs), no hyperparameter search """
    if isinstance(args, dict):
        args = Namespace(**args)

    args.save_path, _ = get_auto_save_path(args)
    early_stop = EarlyStopping(args, verbose=True)

    for epoch in range(int(args.epochs)):
        start_time = time.time()

        train_loss = model.train(train_dataloader)
        results = model.evaluate(valid_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        early_stop(train_loss, results["loss"], results["acc"], model)

        if early_stop.early_stop is True:
            break

        wandb.log(
            dict(
                valid_loss=results["loss"],
                valid_acc=results["acc"],
                train_loss=train_loss,
                learning_rate=model.scheduler.get_last_lr()[0],
            )
        )

        print()

    if_best_score_auto_submit(args, args.save_path)


def debug(args, strategy):
    """ just testing, is it works? """

    if isinstance(args, dict):
        args = Namespace(**args)

    args.batch_size = 32
    args.max_seq_length = 128

    try:
        model, tokenizer = load_model_and_tokenizer(args)
        inputs, labels = load_sample(args, tokenizer)

        preds = model.backbone(**inputs, return_dict=True)
        _ = model.loss_fn(preds.logits, labels)

        query = f"UPDATE STRATEGY SET STATUS = 'RUN' WHERE strategy='{strategy}'"
        execute_query(query)

    except Exception:
        err_message = traceback.format_exc()
        print(err_message)
        query = f"UPDATE STRATEGY SET STATUS = 'PENDING' WHERE strategy='{strategy}'"
        execute_query(query)
        hook_fail_strategy(strategy, err_message)


if __name__ == "__main__":
    args = get_args()
    args = update_args(args, args.strategy, hp_space.strat)
    args = Namespace(**args)

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)

    if args.debug is not True:
        wandb.init(project="p-stage-2")
        wandb.config.update(args)
        wandb.watch(model)
        wandb.run.name = args.base_name

    train_dataloader, valid_dataloader = load_dataloader(args, tokenizer)

    if args.debug is True:
        debug(args, args.strategy)
    else:
        run(args, model, train_dataloader, valid_dataloader)
