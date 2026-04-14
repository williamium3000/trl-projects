from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"



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

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: '{OPSD_DATASET}', '{DAPO_DATASET}'."
        )

    dataset = hf_load_dataset(dataset_name)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
    return split_dataset["train"], split_dataset["test"]
