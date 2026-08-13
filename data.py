# Load MATH and freeze the train/val/test split adapted from math_data_basic_setup.py. Run once only.
# Locking uses MATH train split, so test is independent from training data.
import json
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets

import config

CATEGORIES = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def build():
    all_categories = []
    for category in CATEGORIES:
        data_cat = load_dataset("EleutherAI/hendrycks_math",
                                category, split="test")
        all_categories.append(data_cat)
    dataset = concatenate_datasets(all_categories)

    problems = [{"problem": x["problem"],
                 "solution": x["solution"],
                 "type": x.get("type", ""),
                 "level": x.get("level", "")} for x in dataset]
    print(f"{len(problems)} problems")

    idx = list(range(len(problems)))
    random.Random(config.SEED).shuffle(idx)

    splits = {
        "train": idx[:config.N_TRAIN],
        "val": idx[config.N_TRAIN:config.N_TRAIN + config.N_VAL],
        "test": idx[config.N_TRAIN + config.N_VAL:],
    }

    all_indices = splits["train"] + splits["val"] + splits["test"]
    assert len(all_indices) == len(set(all_indices))
    print({k: len(v) for k, v in splits.items()})

    Path("problems.json").write_text(json.dumps(problems))
    Path("splits.json").write_text(json.dumps(splits))


def load():
    return (json.loads(Path("problems.json").read_text()),
            json.loads(Path("splits.json").read_text()))


def subset(problems, splits, name):
    return [problems[i] for i in splits[name]]


if __name__ == "__main__":
    build()
