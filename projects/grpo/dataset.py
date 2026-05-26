"""Dataset loading for grpo.

Self-contained per repo "trainer self-contained, share by copy" convention.

Set MATH500_EVAL_PATH=data/math500/test.json (relative to repo root) to use
the MATH-500 validation set (industry standard, used by MARTI / SimpleRL-Zoo).
Without this env var, a 150-prompt holdout is carved from the train split.

Co-rewarding-I replication uses two row-aligned parquet files published in the
Co-rewarding GitHub repo (Zhang et al. ICLR 2026, tmlr-group/Co-rewarding):
  - train_original.parquet      (7500 problems, MATH-Train, verl format)
  - train_rewrite_Qwen3-32B.parquet (same 7500 rows positionally, rephrased
                                     via Qwen3-32B; semantically equivalent
                                     question + identical answer)
Both files share schema and are aligned by row index. Override the source dir
via the COREWARDING_DATA_DIR env var (default: ~/research/Co-rewarding/
Co-rewarding-I/data/math). When porting to a pod, set the env var to the NAS
mirror path; nothing else changes.
"""

import json
import os
from pathlib import Path

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"

# Co-rewarding-I replication: paired parquets from tmlr-group/Co-rewarding.
# Selected via these sentinel "dataset names"; actual on-disk path resolved
# via COREWARDING_DATA_DIR (or default) at load time.
COREWARDING_MATH_ORIGINAL = "coreward/math_original"
COREWARDING_MATH_REPHRASED = "coreward/math_rephrased"

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_COREWARDING_DEFAULT_DIR = "~/research/Co-rewarding/Co-rewarding-I/data/math"


def _make_prompt(text):
    return [{"role": "user", "content": f"{text}\n {_INSTRUCTION}"}]


def _load_math500_eval(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list([
        {"prompt": _make_prompt(e["prompt"]), "solution": e["answer"]}
        for e in data
    ])


def _coreward_parquet_path(which: str) -> Path:
    """Resolve Co-rewarding parquet path. `which` ∈ {'original', 'rephrased'}."""
    base = os.environ.get("COREWARDING_DATA_DIR", _COREWARDING_DEFAULT_DIR)
    base = Path(os.path.expanduser(base))
    if which == "original":
        return base / "train_original.parquet"
    elif which == "rephrased":
        return base / "train_rewrite_Qwen3-32B.parquet"
    raise ValueError(f"unknown coreward parquet variant: {which}")


def _load_coreward_parquet(which: str) -> Dataset:
    """Load Co-rewarding verl-format parquet and convert to {prompt, solution}.

    The verl format stores `prompt` as a list of {role, content} dicts (system
    + user); we extract the user content, drop the verl system message (its
    'reason step by step + boxed' instruction is duplicated by our own
    _INSTRUCTION), and wrap via _make_prompt. `solution` is taken from
    `reward_model.ground_truth` (string).

    Row order is preserved exactly so positional index alignment with the
    sibling parquet (e.g. original ↔ rephrased) holds end-to-end.
    """
    import pandas as pd
    parquet_path = _coreward_parquet_path(which)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Co-rewarding parquet not found at {parquet_path}. "
            f"Set COREWARDING_DATA_DIR env var to the directory containing "
            f"train_original.parquet + train_rewrite_Qwen3-32B.parquet. "
            f"(default: {_COREWARDING_DEFAULT_DIR})"
        )
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        # `prompt` is an ndarray of dicts; pull the user message.
        msgs = row["prompt"]
        user_content = None
        for m in msgs:
            if m.get("role") == "user":
                user_content = m["content"]
                break
        if user_content is None:
            raise ValueError(f"no user role in row prompt: {msgs}")
        # `reward_model` is a dict with `ground_truth` (string) + `style`.
        solution = row["reward_model"]["ground_truth"]
        records.append({
            "prompt": _make_prompt(user_content),
            "solution": str(solution),
        })
    return Dataset.from_list(records)


def load_dataset(dataset_name):
    # Co-rewarding-I parquet branch — local file, not HF Hub.
    if dataset_name in (COREWARDING_MATH_ORIGINAL, COREWARDING_MATH_REPHRASED):
        which = "original" if dataset_name == COREWARDING_MATH_ORIGINAL else "rephrased"
        full_train = _load_coreward_parquet(which)
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
            f"'{MATH_LEVEL345_DATASET}', '{MATH_LEVEL12345_DATASET}', "
            f"'{COREWARDING_MATH_ORIGINAL}', '{COREWARDING_MATH_REPHRASED}'."
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
