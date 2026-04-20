"""Dataset loading for co-grpo-dp.

Self-contained per repo "trainer self-contained, share by copy" convention.
Used to importlib-bridge un-grpo-maj/dataset.py; now standalone so changes here
do not affect un-grpo-maj.

Loads one of four math datasets and splits a fixed **150-prompt validation set**
(seed=42) from the HF train portion. This validation set is the inline-eval
target during training (every `--eval_steps`).

Important: for datasets that ship an HF "test" split (MATH-Level345 and
MATH-Level12345), that test split is **not** loaded here. It is reserved for
downstream final-benchmark eval via `eval_benchmarks.sh` (lm-eval-harness on
MATH-500 etc.). Mixing it into inline eval would (a) make the inline eval set
inconsistent across datasets, and (b) leak benchmark data into the training
monitoring loop.
"""

import os

from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"

# Fixed validation split — same size and seed across all four datasets so eval
# numbers are comparable across experiments. Pinned seed (42) means validation
# IDs are identical across runs / restarts / branches.
_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42


def load_dataset(dataset_name):
    """Load (train, validation) for one of the four supported math datasets.

    Args:
        dataset_name (`str`):
            One of OPSD_DATASET, DAPO_DATASET, MATH_LEVEL345_DATASET,
            MATH_LEVEL12345_DATASET.

    Returns:
        `tuple[Dataset, Dataset]`: (train_dataset, validation_dataset).
        Both have columns `{"prompt", "solution"}` only.
        Validation is exactly 150 rows, deterministic across runs.
        Train is HF-train minus those 150, optionally further capped by
        `MAX_SAMPLES` env var (validation is never capped).
    """
    if dataset_name == OPSD_DATASET:
        format_prompt = lambda example: {
            "prompt": f"{example['problem']}\n Please reason step by step, and put your final answer within \\boxed{{}}.",
            "solution": example["Answer"],
        }
    elif dataset_name == DAPO_DATASET:
        format_prompt = lambda example: {
            "prompt": f"{example['prompt']}\n Please reason step by step, and put your final answer within \\boxed{{}}.",
            "solution": example["solution"],
        }
    elif dataset_name in (MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET):
        format_prompt = lambda example: {
            "prompt": f"{example['prompt']}\n Please reason step by step, and put your final answer within \\boxed{{}}.",
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

    # 150-prompt validation holdout from the train portion. test_size accepts
    # an int (absolute count) per HF datasets API. Seed pinned -> deterministic.
    split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        n = min(int(max_samples), len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    return train_dataset, eval_dataset
