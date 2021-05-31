strat = dict()


# 가벼운 모델을 사용하자...

# Just Testing
strat["st00"] = {
    "strategy": "st00",
    "ms_name": "testmodel",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 8,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}

"""
>>> TODO: 데이터셋 비교 실험

st01 : dataset_v1, 전처리 하지 않은 데이터 셋
st02 : dataset_v2, Entity 독립적으로 만든 데이터 셋
st03 : dataset_v3, Entity 독립 + Entity사이 스페이스 없애기
st04 : dataset_v4, 데이터 추가한 후, 전처리 진행하지 않은 것
st05 : dataset_v5, 데이터 추가한 후, dataset_v3 전처리 진행한 것
"""

strat["st01"] = {
    "strategy": "st01",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 64,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}


strat["st02"] = {
    "strategy": "st02",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v2",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 64,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}


strat["st03"] = {
    "strategy": "st03",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v3",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 64,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}


strat["st04"] = {
    "strategy": "st04",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 64,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}


strat["st05"] = {
    "strategy": "st05",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v5",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 64,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
}


"""
>>> TODO: Optimizer, Scheduler 비교

st06 : dataset_v1, Adamw, warm_up
st07 : dataset_v1, Adamw, cyclic
>>> 러닝레이트 증가 후 서서히 감소 VS 증가 감소 반복

st08 : dataset_v1, SGD, cyclic
>>> SGD랑 Cyclic 섞어쓰면 좋다고 해서 사용.

st09 : dataset_v1, Adamp, warm_up
st10 : dataset_v1, Adamp, cyclic
>>> Adamp 성능 한번 테스트 해보기

st11 : dataset_v1, MadGrad, warm_up
st12 : dataset_v1, MadGrad, cyclic
>>> A Momentumized, Adaptive Dual Averaged Gradient Method for Stochastic Optimization
MadGrad 성능 한번 테스트 해보기
"""


strat["st06"] = {
    "strategy": "st06",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "optimizer": "adamw",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 128,
}


strat["st07"] = {
    "strategy": "st07",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "sgdr",
    "scheduler_hp": {
        "first_cycle_steps": 500,
        "cycle_mult": 1.0,
        "warmup_steps": 100,
        "max_lr": 5e-4,
        "min_lr": 5e-5,
        "gamma": 0.5,
    },
    "optimizer": "adamw",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 256,
}


strat["st08"] = {
    "strategy": "st08",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "optimizer": "sgd",
    "optimizer_hp": {"lr": 5e-5, "momentum": 0.3},
    "batch_size": 64,
    "max_seq_length": 128,
}


strat["st09"] = {
    "strategy": "st09",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "optimizer": "adamp",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 128,
}


strat["st10"] = {
    "strategy": "st10",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "sgdr",
    "scheduler_hp": {
        "first_cycle_steps": 500,
        "cycle_mult": 1.0,
        "warmup_steps": 100,
        "max_lr": 5e-4,
        "min_lr": 5e-5,
        "gamma": 0.5,
    },
    "optimizer": "adamp",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 256,
}


strat["st11"] = {
    "strategy": "st11",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "optimizer": "madgrad",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 128,
}


strat["st12"] = {
    "strategy": "st12",
    "ms_name": "koelecv3",
    "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
    "data_kind": "dataset_v1",
    "scheduler": "sgdr",
    "scheduler_hp": {
        "first_cycle_steps": 500,
        "cycle_mult": 1.0,
        "warmup_steps": 100,
        "max_lr": 5e-4,
        "min_lr": 5e-5,
        "gamma": 0.5,
    },
    "optimizer": "madgrad",
    "optimizer_hp": {"lr": 5e-5, "eps": 1e-8},
    "batch_size": 64,
    "max_seq_length": 256,
}


"""
>>> TODO: 효과가 좋았던 기법들을 사용하여 모델 사이즈 증가하기, sampler 사용/미사용 둘 다.
>>> dataset_v4, adamw, warm_up, 128

st13: xlm-roberta, use sampler
st14: xlm-roberta, not use sampler

st15: kobert, use sampler
st16: kobert, not use sampler

st17: distilkobert, use sampler
st18: distilkobert, not use sampler

st19: rbert, use sampler
st20: rbert, not use sampler

"""

strat["st13"] = {
    "strategy": "st13",
    "ms_name": "xlmroberta",
    "model_name_or_path": "xlm-roberta-large",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": True,
}

strat["st14"] = {
    "strategy": "st14",
    "ms_name": "xlmroberta",
    "model_name_or_path": "xlm-roberta-large",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": False,
}


strat["st15"] = {
    "strategy": "st15",
    "ms_name": "kobert",
    "model_name_or_path": "monologg/kobert",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": True,
}

strat["st16"] = {
    "strategy": "st16",
    "ms_name": "kobert",
    "model_name_or_path": "monologg/kobert",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": False,
}

strat["st17"] = {
    "strategy": "st17",
    "ms_name": "distilkobert",
    "model_name_or_path": "monologg/distilkobert",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": True,
}

strat["st18"] = {
    "strategy": "st18",
    "ms_name": "distilkobert",
    "model_name_or_path": "monologg/distilkobert",
    "data_kind": "dataset_v4",
    "optimizer": "adamw",
    "scheduler": "warm_up",
    "scheduler_hp": {"num_warmup_steps": 500},
    "batch_size": 32,
    "max_seq_length": 128,
    "optimizer_hp": {"lr": 1e-5, "eps": 1e-8},
    "use_sampler": False,
}
