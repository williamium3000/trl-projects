from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"



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
            f"Unsupported dataset '{dataset_name}'. Supported datasets: '{OPSD_DATASET}', '{DAPO_DATASET}', '{MATH_LEVEL345_DATASET}', '{MATH_LEVEL12345_DATASET}'."
        )

    dataset = hf_load_dataset(dataset_name)
    if "test" in dataset:
        train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)
        test_dataset = dataset["test"].map(format_prompt, remove_columns=dataset["test"].column_names)
        return train_dataset, test_dataset
    else:
        train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)
        split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
        return split_dataset["train"], split_dataset["test"]
