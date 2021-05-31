import os
import copy
import pickle
from functools import partial

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_org_dataset(dataset_dir):
    train_df = pd.read_csv(
        os.path.join(dataset_dir, "train.tsv"), delimiter="\t", header=None
    )
    test_df = pd.read_csv(
        os.path.join(dataset_dir, "test.tsv"), delimiter="\t", header=None
    )
    return train_df, test_df


def modify_missing_label(train_df):
    mislabel = """wikitree-55837-4-0-2-10-11
wikitree-62775-3-3-7-0-2
wikitree-12599-4-108-111-4-7
wikipedia-25967-115-24-26-35-37
wikipedia-16427-6-14-17-20-22
wikipedia-16427-8-0-3-26-28
wikitree-19765-5-30-33-6-8
wikitree-58702-0-18-20-22-24
wikitree-71638-8-21-23-15-17"""
    mislabel = mislabel.split("\n")

    modlabel = [
        "단체:구성원",
        "단체:본사_도시",
        "관계_없음",
        "관계_없음",
        "관계_없음",
        "관계_없음",
        "관계_없음",
        "관계_없음",
        "관계_없음",
    ]

    assert len(mislabel) == len(modlabel)

    for i, rows in train_df.iterrows():
        if rows[0] not in mislabel:
            continue

        idx = mislabel.index(rows[0])
        print(f"{train_df.iloc[i][8]} -> {modlabel[idx]}")
        train_df.iat[i, 8] = modlabel[idx]

    return train_df


def addition_one_label(train_df, data_dir):
    """ 인물:사망_국가 레이블 1개 추가 """
    words = "내부 마케도니아 혁명 기구의 회원으로 활동했으며 1934년에는 프랑스 마르세유에서 유고슬라비아 왕국의 알렉산다르 1세 국왕과 프랑스의 루이 바르투 외무장관을 암살했다."
    temp = pd.Series(
        [
            "wikipedia-13641-1-57-64-35-42",
            words,
            "알렉산다르 1세",
            57,
            64,
            "프랑스",
            35,
            37,
            "인물:사망_국가",
        ]
    )

    train_df = train_df.append(temp, ignore_index=True)
    return train_df


def make_sub_dataset(sub_dataset, label_type, words_trans_fn, save_path):
    datas = {}
    e1s, e2s, words, labels = [], [], [], []

    for i, rows in sub_dataset.iterrows():
        word, e1, e2 = words_trans_fn(rows[1], rows[2], rows[5])
        words.append(word)
        e1s.append(e1)
        e2s.append(e2)

        if rows[8] == "blind":
            labels.append(100)
        else:
            labels.append(label_type[rows[8]])

    datas["e1"] = e1s
    datas["e2"] = e2s
    datas["words"] = words
    datas["labels"] = labels

    with open(save_path, "wb") as f:
        pickle.dump(datas, f)


def make_dataset(
    train_dataset, test_dataset, data_dir, data_kind="dataset_v1", words_trans_fn=None
):

    save_path = os.path.join(data_dir, data_kind)
    label_path = os.path.join(data_dir, "label_type.pkl")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print(f"Already {save_path} exists")
        return

    split = StratifiedShuffleSplit(n_splits=5, test_size=0.15)  # 고정
    split_key = 8

    with open(label_path, "rb") as f:
        label_type = pickle.load(f)

    for i, (t_idx, v_idx) in enumerate(
        split.split(train_dataset, train_dataset[split_key])
    ):
        train_sub_dataset = train_dataset.loc[t_idx]
        valid_sub_dataset = train_dataset.loc[v_idx]

        assert len(set(t_idx).intersection(set(v_idx))) == 0

        path = os.path.join(save_path, f"train-{i}.pkl")
        make_sub_dataset(train_sub_dataset, label_type, words_trans_fn, path)

        path = os.path.join(save_path, f"valid-{i}.pkl")
        make_sub_dataset(valid_sub_dataset, label_type, words_trans_fn, path)

    path = os.path.join(save_path, "test.pkl")
    make_sub_dataset(test_dataset, label_type, words_trans_fn, path)

    show_sample_dataset(data_dir, data_kind)


def make_entity_independent(word, e1, e2):
    """Entity 주변에 스페이스를 둠으로써 단어 해결
    >>> Entity: '배우', Word: '여배우는' -> Return: '여 배우 는'
    """
    new_word = word.replace(e1, " " + e1 + " ")
    new_word = new_word.replace(e2, " " + e2 + " ")
    new_word = new_word.replace("  ", " ").strip()
    return new_word, e1, e2


def make_entity_to_one_words(word, e1, e2):
    """Entity 사이의 Space를 제거한다.
    >>> Entity: '랜드 로버', Word: '랜드 로버는' -> Return: '랜드로버는'
    """
    new_e1 = e1.replace(" ", "")
    new_e2 = e2.replace(" ", "")

    new_word = word.replace(e1, new_e1)
    new_word = new_word.replace(e2, new_e2)

    return new_word, new_e1, new_e2


def show_sample_dataset(data_dir, data_kind):
    train_path = os.path.join(data_dir, data_kind, "train-1.pkl")
    valid_path = os.path.join(data_dir, data_kind, "valid-1.pkl")
    test_path = os.path.join(data_dir, data_kind, "test.pkl")

    with open(train_path, "rb") as f:
        train = pickle.load(f)

    with open(valid_path, "rb") as f:
        valid = pickle.load(f)

    with open(test_path, "rb") as f:
        test = pickle.load(f)

    msg_fotmat = "Entity01: {}\nEntity02: {}\nWord: {}"

    print(f"Dataset_kind: {data_kind}")

    for kind, dataset in zip(["train", "valid", "test"], [train, valid, test]):
        print(kind.upper())
        print(
            msg_fotmat.format(dataset["e1"][0], dataset["e2"][0], dataset["words"][0])
        )
        print()

    print()


def preprocess_dataset(data_dir, df, sample_num=200):
    datas = {i: [] for i in range(9)}

    with open(os.path.join(data_dir, "label_type.pkl"), "rb") as f:
        label_type = pickle.load(f)

    rev = {v: k for k, v in label_type.items()}

    for i, row in df.sample(sample_num).iterrows():
        word, e1, e2, label = row[0], row[1], row[2], row[3]
        datas[0].append("wikipedia")
        datas[1].append(word)
        datas[2].append(e1)
        datas[3].append(0)
        datas[4].append(0)
        datas[5].append(e2)
        datas[6].append(0)
        datas[7].append(0)
        datas[8].append(rev[label])

    return datas


def add_dataset_with_new_data(train_dataset, data_dir):
    filenames = [
        "19_bornIn_city",
        "26_bornIn_country",
        "37_dienIn_city",
        "40_dienIn_country",
    ]

    for filename in filenames:
        temp_df = pd.read_csv(os.path.join(data_dir, filename + ".tsv"), delimiter="\t")

        datas = preprocess_dataset(data_dir, temp_df)
        datas = pd.DataFrame(datas)

        train_dataset = train_dataset.append(datas, ignore_index=True)

    filename = "all_csv.tsv"

    new_dataset = pd.read_csv(os.path.join(path, filename), delimiter="\t", header=None)

    # sample 100
    for k in set(new_dataset[8]):
        temp = new_dataset[new_dataset[8] == k].sample(100)
        train_dataset = train_dataset.append(temp, ignore_index=True)

    return train_dataset


def main(data_dir):
    # Fill Missing Label, Low Class Label
    # Not Valid, It's Test!
    train_dataset, test_dataset = load_org_dataset(data_dir)
    train_dataset = modify_missing_label(train_dataset)
    train_dataset = addition_one_label(train_dataset, data_dir)

    def dataset_v1_pipeline(word, e1, e2):
        return word, e1, e2

    def dataset_v2_pipeline(word, e1, e2):
        word, e1, e2 = make_entity_independent(word, e1, e2)
        return word, e1, e2

    def dataset_v3_pipeline(word, e1, e2):
        word, e1, e2 = make_entity_independent(word, e1, e2)
        word, e1, e2 = make_entity_to_one_words(word, e1, e2)
        return word, e1, e2

    make_dataset_ = partial(
        make_dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        data_dir=data_dir,
    )

    make_dataset_(data_kind="dataset_v1", words_trans_fn=dataset_v1_pipeline)
    make_dataset_(data_kind="dataset_v2", words_trans_fn=dataset_v2_pipeline)
    make_dataset_(data_kind="dataset_v3", words_trans_fn=dataset_v3_pipeline)

    train_dataset = add_dataset_with_new_data(copy.deepcopy(train_dataset), data_dir)

    make_dataset_ = partial(
        make_dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        data_dir=data_dir,
    )

    make_dataset_(data_kind="dataset_v4", words_trans_fn=dataset_v1_pipeline)
    make_dataset_(data_kind="dataset_v5", words_trans_fn=dataset_v3_pipeline)


if __name__ == "__main__":
    path = "/home/j-gunmo/storage/data/input/data/"
    server_path = "/opt/ml/input/data/"

    if os.path.abspath(".").split("/")[1] == "opt":
        path = server_path

    main(path)
