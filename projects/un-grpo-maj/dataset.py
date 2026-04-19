import os

from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"

# Datasets that already provide their own train/test splits (skip extra split).
_PRESPLIT_DATASETS = {MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET}


def load_dataset(dataset_name):
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

    if dataset_name in _PRESPLIT_DATASETS:
        train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)
        eval_dataset = dataset["test"].map(format_prompt, remove_columns=dataset["test"].column_names)
    else:
        train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)
        split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
        train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        n = min(int(max_samples), len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    return train_dataset, eval_dataset
