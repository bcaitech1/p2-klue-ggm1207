import os
import traceback
from importlib import reload
from argparse import Namespace

import torch
import wandb
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.wandb import wandb_mixin

import hp_space
from config import get_args
from train import debug, run
from prepare import load_dataloader
from database import sample_strategy
from networks import load_model_and_tokenizer
from inference import if_best_score_auto_submit
from slack import hook_simple_text, hook_fail_ray
from utils import update_args, EarlyStopping, set_seed


class CustomStopper(tune.Stopper):
    def __init__(self, args):
        if isinstance(args, dict):
            args = Namespace(**args)

        self.should_stop = False
        self.args = args

    def __call__(self, trial_id, result):
        if not self.should_stop and result["valid_acc"] > 0.85:
            self.should_stop = True

        return self.should_stop or result["training_iteration"] >= self.args.epochs

    def stop_all(self):
        return self.should_stop


@wandb_mixin
def main(config, checkpoint_dir=None):
    step = 0
    args = Namespace(**config)

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    train_dataloader, valid_dataloader = load_dataloader(args, tokenizer)
    es_helper = EarlyStopping(args, verbose=True)  # Use For Ensemble

    # Tune can automatically and periodically save/checkpoint your model.
    if checkpoint_dir is not None:  # Use For PBT
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model"])
        model.optimizer.load_state_dict(checkpoint["optim"])
        step = checkpoint["step"]

    while True:
        train_loss = model.train(train_dataloader)
        results = model.evaluate(valid_dataloader)

        es_helper(train_loss, results["loss"], results["acc"], model)

        # wandb.log는 tune.report, tune.checkpoint_dir 보다 선행 되어야 한다.
        wandb.log(
            dict(
                valid_loss=results["loss"],
                valid_acc=results["acc"],
                train_loss=train_loss,
                learning_rate=model.scheduler.get_last_lr()[0],
            )
        )

        step += 1

        # 뭔지 모르겠지만 여기서 걍 끝남.
        tune.report(valid_loss=results["loss"], valid_acc=results["acc"], train_loss=train_loss)

        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save({"model": model.state_dict(), "optim": model.optimizer.state_dict(), "step": step}, path)


def run_with_raytune(args):
    """ 하이퍼파라미터 설정하는 곳 """
    if isinstance(args, Namespace):
        args = vars(args)  # Namespace to dict

    while True:
        reload(hp_space)
        strategy, status, _, _ = sample_strategy()

        # update hp_space, dataset_idx, wandb, save_path, base_name
        args = update_args(args, strategy, hp_space.strat)

        if status == "READY":  # if status == "READY" then Check pipeline
            debug(args, strategy)
            torch.cuda.empty_cache()  # Debug 이후에 할당된 메모리 해제
            continue

        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=1,
            hyperparam_mutations={  # use when explore
                "optimizer_hp": {"lr": tune.uniform(0.0001, 0.0005)},
                "weight_decay": tune.uniform(0.001, 0.02),
            },
        )

        hook_simple_text(f":pray: {args['base_name']} PBT 시작합니다!!")
        stopper = CustomStopper(args)

        tune.run(
            main,
            name="pbt_test",
            mode="min",
            stop=stopper,
            num_samples=4,
            metric="valid_loss",
            scheduler=scheduler,
            keep_checkpoints_num=3,
            checkpoint_score_attr="valid_loss",  # valid loss 기준으로 저장.
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=args,
        )

        torch.cuda.empty_cache()

        hook_simple_text(f":joy: {args['base_name']} 학습 끝!!!")

        if_best_score_auto_submit(args, args["save_path"])

        torch.cuda.empty_cache()


def run_without_raytune():
    while True:
        reload(hp_space)

        args = get_args()  # default
        strategy, status, _, _ = sample_strategy()
        args = update_args(args, strategy, hp_space.strat)  # strategy
        args = Namespace(**args)

        if status == "READY":  # if status == "READY" then Check pipeline
            debug(args, strategy)
            torch.cuda.empty_cache()  # Debug 이후에 할당된 메모리 해제
            continue

        print("args:", args)

        wandb.init(project="p-stage-2", reinit=True)
        wandb.run.name = args.base_name
        model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)

        wandb.watch(model)
        wandb.config.update(args)

        train_dataloader, valid_dataloader = load_dataloader(args, tokenizer)

        hook_simple_text(f":pray: {args.base_name} PBT 시작합니다!!")

        run(args, model, train_dataloader, valid_dataloader)

        torch.cuda.empty_cache()
        hook_simple_text(f":joy: {args.base_name} 학습 끝!!!")


if __name__ == "__main__":
    try:
        run_without_raytune()
    except Exception:
        err_message = traceback.format_exc()
        print(err_message)
        hook_fail_ray(err_message)
