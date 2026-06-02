"""Dataset loading for un-grpo-maj.

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

# CoMAS replication: local json (data/comas/), not HF Hub. `blended` = 5000
# (2000 science + 1500 coding + 1500 math); `math` = the 1500 math subset.
# Reward task-routed (see self-label trainer / reward): math/science → boxed+
# sympy; coding → run-output self-majority (unsupervised, mirrors CoMAS).
COMAS_BLENDED = "comas/blended"
COMAS_MATH = "comas/math"

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_COMAS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "comas")


def _make_prompt(text):
    return [{"role": "user", "content": f"{text}\n {_INSTRUCTION}"}]


def _load_math500_eval(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list([
        {"prompt": _make_prompt(e["prompt"]), "solution": e["answer"]}
        for e in data
    ])


def _load_comas_json(which: str) -> Dataset:
    """Load CoMAS data (json list) -> columns {prompt, solution, task, test_code}.

    `which` ∈ {'blended', 'math'}. `task` routes the reward; `test_code` is a
    SEPARATE persistent column (coding asserts) that must survive the self-label
    `solution` overwrite to reach the reward fn. Empty string for non-coding rows.
    """
    base = os.environ.get("COMAS_DATA_DIR", _COMAS_DATA_DIR)
    fname = {"blended": "blended_train.json", "math": "math_train.json"}[which]
    path = os.path.join(os.path.expanduser(base), fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CoMAS data not found at {path}. Expected blended_train.json / "
            f"math_train.json under data/comas/ (or set COMAS_DATA_DIR)."
        )
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list([
        {
            "prompt": _make_prompt(e["prompt"]),
            "solution": str(e["answer"]),
            "task": e.get("task", "math"),
            "test_code": e["answer"] if e.get("task") == "coding" else "",
        }
        for e in data
    ])


def load_dataset(dataset_name):
    # CoMAS data branch — local json, not HF Hub (returns task/test_code columns).
    if dataset_name in (COMAS_BLENDED, COMAS_MATH):
        which = "blended" if dataset_name == COMAS_BLENDED else "math"
        full_train = _load_comas_json(which)
        split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
        train_dataset, eval_dataset = split["train"], split["test"]
        max_samples = os.environ.get("MAX_SAMPLES")
        if max_samples is not None:
            train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))
        math500_path = os.environ.get("MATH500_EVAL_PATH")
        if math500_path is not None:
            eval_dataset = _load_math500_eval(math500_path)
        return train_dataset, eval_dataset

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
