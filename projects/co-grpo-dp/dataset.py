"""Dataset loading for co-grpo-dp.

Self-contained per repo "trainer self-contained, share by copy" convention.

Set MATH500_EVAL_PATH=data/math500/test.json (relative to repo root) to use
the MATH-500 validation set (industry standard, used by MARTI / SimpleRL-Zoo).
Without this env var, a 150-prompt holdout is carved from the train split.
"""

import json
import os

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _make_prompt(text):
    return [{"role": "user", "content": f"{text}\n {_INSTRUCTION}"}]


def _load_math500_eval(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list([
        {"prompt": _make_prompt(e["prompt"]), "solution": e["answer"]}
        for e in data
    ])


def load_dataset(dataset_name):
    if dataset_name == OPSD_DATASET:
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["problem"]),
            "solution": example["Answer"],
        }
    elif dataset_name == DAPO_DATASET:
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["prompt"]),
            "solution": example["solution"],
        }
    elif dataset_name in (MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET):
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["prompt"]),
            "solution": example["answer"],
        }
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: "
            f"'{OPSD_DATASET}', '{DAPO_DATASET}', "
            f"'{MATH_LEVEL345_DATASET}', '{MATH_LEVEL12345_DATASET}'."
        )

    dataset = hf_load_dataset(dataset_name)
    full_train = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)

    split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        n = min(int(max_samples), len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    math500_path = os.environ.get("MATH500_EVAL_PATH")
    if math500_path is not None:
        eval_dataset = _load_math500_eval(math500_path)

    return train_dataset, eval_dataset
