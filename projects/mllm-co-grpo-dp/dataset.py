"""Dataset loading for mllm-co-grpo-dp.

Multimodal sibling of `co-grpo-dp/dataset.py`. Each example carries:

- `prompt` (`list[dict]`): R1-V chat-format prompt, **no system role**,
  question + suffix in a single user content string.
- `image` (`PIL.Image`): the input image. TRL GRPOTrainer's collate
  passes this through `AutoProcessor` for VLM tokenization.
- `solution` (`str`): ground-truth answer (used by reward path in eval
  mode and by `co_labeling/oracle_accuracy_me` diagnostic in train mode).

R1-V baseline prompt template (memory `mllm_co_grpo_dp_plan` D1):

    "{Question} Output the thinking process in <think> </think> and
    final answer in <answer> </answer> tags."

The suffix is appended to the raw question text. No system message —
R1-V uses prompt-suffix instruction injection.

Per memory D3, one source per launch via `--train_dataset`. Do not concat.

Eval set is loaded from a **fixed R1-V test set** (not carved from train)
so inline accuracy is directly comparable to R1-V baseline numbers:
  - CLEVR-Counting → SuperCLEVR-200
  - GEOQA          → GeoQA-Test-Direct-Answer-735

Set `MLLM_EVAL_PATH=path/to/eval.jsonl` to point at a local jsonl eval
file (one `{"problem", "image", "solution"}` per line, `image` is a path
relative to MLLM_EVAL_IMAGE_DIR). Without this env var the loader carves
a 150-prompt holdout from train (seed 42) — useful for dry-run / sanity
only; replace with the real eval set before any reportable number.
"""

import json
import os
from pathlib import Path

from PIL import Image
from datasets import Dataset
from datasets import Image as HFImage
from datasets import load_dataset as hf_load_dataset

from verifiers.math_verify_wrapper import extract_answer_tag


# Training datasets (HuggingFace Hub IDs)
CLEVR_COUNTING_DATASET = "leonardPKU/clevr_cogen_a_train"
GEOQA_DATASET = "leonardPKU/GEOQA_R1V_Train_8K"

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_PROMPT_SUFFIX = (
    " Output the thinking process in <think> </think> and final answer in "
    "<answer> </answer> tags."
)


def _make_prompt(question_text):
    """R1-V style prompt: no system role, multimodal user content (image + text).

    Content **must** be a list with an explicit `{"type": "image"}` part — both
    Qwen2.5-VL and InternVL3.5 chat templates branch on
    `message['content'] is string`:
      - string content → text is rendered as-is, **no image placeholder emitted**
      - list content   → each `{"type": "image"}` part emits the model's image
        placeholder token(s), required for vLLM mm processing and model forward.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question_text}{_PROMPT_SUFFIX}"},
        ],
    }]


def _convert_to_rgb(example):
    """Ensure image is RGB. CLEVR/GEOQA are RGB but be defensive at the boundary."""
    img = example["image"]
    if img.mode != "RGB":
        example["image"] = img.convert("RGB")
    return example


def _load_local_eval_jsonl(jsonl_path, image_dir):
    """Load a fixed eval set from a local jsonl + image directory.

    Each line is `{"problem": str, "image": <path relative to image_dir>,
    "solution": str}`. Returns a Dataset with `prompt` / `image` / `solution`
    columns, ready for inline eval. Images are loaded as PIL eagerly so
    the eval iterator doesn't pay disk cost on every step.
    """
    image_dir = Path(image_dir) if image_dir is not None else None
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row["image"]
            if image_dir is not None and not os.path.isabs(img_path):
                img_path = image_dir / img_path
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im.load()
            records.append({
                "prompt": _make_prompt(row["problem"]),
                "image": im,
                "solution": row["solution"],
            })
    return Dataset.from_list(records)


def load_dataset(dataset_name):
    """Load train + eval datasets for the given source name.

    Args:
        dataset_name (`str`): one of `CLEVR_COUNTING_DATASET` or `GEOQA_DATASET`.

    Returns:
        `tuple[Dataset, Dataset]`: `(train, eval)`. Both have columns
        `prompt` (chat list), `image` (PIL), `solution` (str).

    Env vars (optional):
        MLLM_EVAL_PATH:
            Path to a jsonl eval file. Schema: one
            `{"problem": str, "image": <path>, "solution": str}` per line.
            If unset, eval is carved from train (size 150, seed 42).
        MLLM_EVAL_IMAGE_DIR:
            Directory that `image` paths in MLLM_EVAL_PATH are relative
            to. Ignored if `image` paths are already absolute.
        MAX_SAMPLES:
            Truncate train set to first N examples (debug / sanity only).
    """
    if dataset_name not in (CLEVR_COUNTING_DATASET, GEOQA_DATASET):
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: "
            f"'{CLEVR_COUNTING_DATASET}', '{GEOQA_DATASET}'."
        )

    raw = hf_load_dataset(dataset_name)
    train_split = raw["train"]

    columns = set(train_split.column_names)
    # R1-V datasets standardize on `problem` (question text) + `solution`
    # (answer string) + `image` (PIL via HFImage feature). Fail fast if
    # schema doesn't match — see dataset card on HF.
    if "image" not in columns:
        raise ValueError(
            f"Dataset '{dataset_name}' has no `image` column. Columns: {columns}"
        )
    question_col = "problem" if "problem" in columns else None
    answer_col = "solution" if "solution" in columns else None
    if question_col is None or answer_col is None:
        raise ValueError(
            f"Dataset '{dataset_name}' must have 'problem' and 'solution' "
            f"columns. Found columns: {columns}"
        )

    def _format(example):
        # CLEVR/GEOQA store solution as '<answer> X </answer>'. Strip the wrapper
        # so `solution` is the bare gold (e.g. '3', '145°') — matches what
        # reward_correctness extracts from completions, and avoids relying on
        # math_verify's tolerance of embedded answer tags.
        raw_sol = str(example[answer_col])
        stripped = extract_answer_tag(raw_sol)
        return {
            "prompt": _make_prompt(example[question_col]),
            "image": example["image"],
            "solution": stripped if stripped is not None else raw_sol.strip(),
        }

    full_train = train_split.map(_format, remove_columns=train_split.column_names)
    # Ensure `image` column is decoded to PIL (no-op if already PIL).
    if not isinstance(full_train.features["image"], HFImage):
        full_train = full_train.cast_column("image", HFImage())
    full_train = full_train.map(_convert_to_rgb)

    # Train / eval split: carve 150 holdout (seed 42) by default.
    split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        n = min(int(max_samples), len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    eval_path = os.environ.get("MLLM_EVAL_PATH")
    if eval_path is not None:
        eval_image_dir = os.environ.get("MLLM_EVAL_IMAGE_DIR")
        eval_dataset = _load_local_eval_jsonl(eval_path, eval_image_dir)

    return train_dataset, eval_dataset


__all__ = [
    "CLEVR_COUNTING_DATASET",
    "GEOQA_DATASET",
    "load_dataset",
]
