import os
import re
import json
import requests

import torch
import numpy as np
import pandas as pd

from networks import load_model_and_tokenizer
from prepare import load_dataloader, load_test_dataloader
from database import update_strategy_statistics, insert_model_scores


def auto_submit(user_key, description, file_path):
    url = (
        f"http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/4/presigned_url/?description={description}"
        + "&hyperparameters={%22training%22:{},%22inference%22:{}}"
    )

    headers = {"Authorization": user_key}
    res = requests.get(url, headers=headers)
    data = json.loads(res.text)
    submit_url = data["url"]

    body = {
        "key": "app/Competitions/000004/Users/{}/Submissions/{}/output.csv".format(
            str(data["submission"]["user"]).zfill(8),
            str(data["submission"]["local_id"]).zfill(4),
        ),
        "x-amz-algorithm": data["fields"]["x-amz-algorithm"],
        "x-amz-credential": data["fields"]["x-amz-credential"],
        "x-amz-date": data["fields"]["x-amz-date"],
        "policy": data["fields"]["policy"],
        "x-amz-signature": data["fields"]["x-amz-signature"],
    }

    requests.post(url=submit_url, data=body, files={"file": open(file_path, "rb")})


def if_best_score_auto_submit(args, save_path):
    """ 경로만 있으면 된다. """

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    _, valid_dataloader = load_dataloader(args, tokenizer)

    model.load_state_dict(torch.load(save_path))

    results = model.evaluate(valid_dataloader, return_keys=["acc"])

    s_cnt = re.findall(r"[\d]{3}", save_path)[-1]

    is_submit = update_strategy_statistics(args, results["acc"])
    insert_model_scores(args, s_cnt, results["acc"])

    if (is_submit is True) and args.auto_sub:
        submission_inference(args, model, tokenizer, save_path)


def submission_inference(args, model, tokenizer, save_path):
    """ save_path: .../st01_kobert_002.pth for parsing """
    model.to(args.device)
    torch.cuda.empty_cache()

    test_dataloader = load_test_dataloader(args, tokenizer)

    base_name = os.path.basename(save_path)[:-4]
    save_path = os.path.join(args.submit_dir, base_name) + ".csv"

    results = model.evaluate(test_dataloader, return_keys=["preds", "logits"])

    print(results)

    #  preds = np.array(results["preds"]).reshape(-1)
    #  output = pd.DataFrame(preds, columns=["pred"])
    #  output.to_csv(save_path, index=False)

    user_key = "Bearer 5c12695179ea1f0a97aec9ce2be8da028755f095"
    #  auto_submit(user_key, base_name, save_path)


if __name__ == "__main__":
    import hp_space
    from config import get_args
    from utils import update_args
    from argparse import Namespace
    from prepare import load_sample

    #  args = get_args()
    #  args = update_args(args, args.strategy, hp_space.strat)
    #  args = Namespace(**args)
    #
    #  save_path = (
    #      "/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/weights/st00_testmodel_001.pth",
    #  )
    #
    #  print(args)
    #
    #  #  if_best_score_auto_submit(
    #  #      args,
    #  #      save_path
    #  #  )
    #
    #  model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    #  test_dataloader = load_test_dataloader(args, tokenizer)
    #
    #  inputs, samples = load_sample(args, tokenizer, is_train=False)
    #
    #  results = model.evaluate(test_dataloader, return_keys=["acc", "logits"])
    #  print(results)

    user_key = "Bearer 5c12695179ea1f0a97aec9ce2be8da028755f095"

    auto_submit(user_key, "test", "ensemble6.csv")
