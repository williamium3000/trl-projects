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

# Additional MLLM training datasets. Each spec maps the source schema → our
# {prompt, image, solution}. Field names verified from the HF dataset viewer
# (2026-06-01). Sentinels: question="@chat" reads the chat-list `prompt` column
# (OpenMMReasoner); answer="@reward" reads reward_model["ground_truth"].
# MMFineReason is one repo with two splits (rl / sft) — the `#sft` key selects
# the sft split via spec["hf_id"]. "<image>" placeholders in question text are
# stripped (the prompt builder injects the image part separately).
ZWZ_37K = "williamium/zwz-37k"
MMFINEREASON = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
MMFINEREASON_SFT = MMFINEREASON + "#sft"
OPENMMREASONER = "OpenMMReasoner/OpenMMReasoner-RL-74K"
OPEN_R1_8K = "lmms-lab/multimodal-open-r1-8k-verified"
GEOMETRY3K = "hiyouga/geometry3k"
MMR1_MATH = "MMR1/MMR1-Math-RL-Data-v0"

_OPENMMR_SUBSETS = ["virl39k", "thinklite_vl_hard", "tqa_train",
                    "wemath_standard", "mmk12", "wemath_pro", "algopuzzle"]

_SPECS = {
    ZWZ_37K:          dict(subset="37k", split="train", image="images", question="problem", answer="answer"),
    MMFINEREASON:     dict(split="rl",   image="image",  question="question", answer="answer", mmfr_filter=True),
    MMFINEREASON_SFT: dict(hf_id=MMFINEREASON, split="sft", image="image", question="question", answer="answer"),
    OPENMMREASONER:   dict(concat=_OPENMMR_SUBSETS, split="train", image="images", question="@chat", answer="@reward"),
    OPEN_R1_8K:       dict(split="train", image="image",  question="problem", answer="@sol_answer"),
    GEOMETRY3K:       dict(split="train", image="images", question="problem", answer="answer"),
    MMR1_MATH:        dict(split="train", image="images", question="problem", answer="answer"),
}

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


# Cap the longest image side so Qwen2.5-VL's native dynamic-resolution tiling
# can't blow past vllm_max_model_length. A 1514x720 MathVista figure yields
# ~11k image tokens uncapped (crashes vLLM at max_model_len); capping the long
# side to 1024 bounds it to ~1.4k tokens. No-op for InternVL (forced to 1 tile)
# and for already-small GeoQA diagrams.
_MAX_LONG_SIDE = 1024


def _cap_image(img):
    """RGB + downscale so max(w, h) <= _MAX_LONG_SIDE (preserves aspect ratio)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / long_side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    return img


def _convert_to_rgb(example):
    """Ensure image is RGB and capped to `_MAX_LONG_SIDE` on the long edge."""
    example["image"] = _cap_image(example["image"])
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
                im.load()
                im = _cap_image(im)
            records.append({
                "prompt": _make_prompt(row["problem"]),
                "image": im,
                "solution": row["solution"],
            })
    return Dataset.from_list(records)


def _extract_chat_text(prompt):
    """Pull the user-turn text out of a chat-list `prompt` (OpenMMReasoner).

    `content` may be a plain string or a list of `{type, text}` parts. Returns
    the concatenated user text with any `<image>` placeholder stripped.
    """
    parts = []
    for msg in prompt:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("text"):
                    parts.append(p["text"])
    return " ".join(parts).replace("<image>", "").strip()


def _load_spec_dataset(dataset_name):
    """Load + normalize one of the `_SPECS` datasets → {prompt, image, solution}."""
    spec = _SPECS[dataset_name]
    hf_id = spec.get("hf_id", dataset_name)
    split = spec.get("split", "train")

    if "concat" in spec:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(
            [hf_load_dataset(hf_id, name=s, split=split) for s in spec["concat"]]
        )
    elif "subset" in spec:
        ds = hf_load_dataset(hf_id, name=spec["subset"], split=split)
    else:
        ds = hf_load_dataset(hf_id, split=split)

    if spec.get("mmfr_filter"):
        # RL split: keep self-consistent, verifiable, non-degenerate items
        # (both judges agree, and 0 < pass_rate < 1 so reward has variance).
        ds = ds.filter(lambda r: r["is_consistent"] and 0.0 < r["pass_rate"] < 1.0)

    img_f, q_f, a_f = spec["image"], spec["question"], spec["answer"]

    def _fmt(ex):
        img = ex[img_f]
        if isinstance(img, list):
            img = img[0]
        # Cap BEFORE the map writes to Arrow. Full-res zwz images (~2MB each) blow
        # past pyarrow's 2GB int32 offset limit once writer_batch_size=1000 rows of
        # image bytes are combined into a single shard (ArrowInvalid: offset overflow,
        # crashes at Map 999). Capping here keeps each image ~200-500KB; also speeds
        # the map and shrinks the on-disk cache. (Distinct from the column prune above,
        # which fixes CPU-RAM OOM — this fixes the Arrow write-size limit.)
        img = _cap_image(img)
        if q_f == "@chat":
            question = _extract_chat_text(ex["prompt"])
        else:
            question = str(ex[q_f]).replace("<image>", "").strip()
        if a_f == "@reward":
            ans = ex["reward_model"]["ground_truth"]
        elif a_f == "@sol_answer":
            # open-r1: `original_answer` is prose; the clean gold lives in the
            # `solution` field's <answer>...</answer> tag. Fall back to the raw
            # original_answer only if no tag is present.
            ans = extract_answer_tag(ex["solution"]) or ex.get("original_answer", "")
        else:
            ans = ex[a_f]
        return {"prompt": _make_prompt(question), "image": img, "solution": str(ans).strip()}

    # Prune columns the formatter never reads BEFORE the map. zwz carries a SECOND
    # full-res image column (`original_images`, 2250x1500) + bbox/extra_info; left
    # in, every row decodes ~2x the images, and 8 DDP ranks mapping 37k pairs on
    # one node exhaust CPU RAM → OOM-killed mid-map with no traceback (2026-06-03).
    # Keep only what `_fmt` reads.
    _keep = {img_f, "prompt" if q_f == "@chat" else q_f}
    if a_f == "@reward":
        _keep.add("reward_model")
    elif a_f == "@sol_answer":
        _keep.update(("solution", "original_answer"))
    else:
        _keep.add(a_f)
    ds = ds.remove_columns([c for c in ds.column_names if c not in _keep])

    # MAX_SAMPLES truncates BEFORE the (image-heavy) map so debug/sanity runs on
    # huge sources (e.g. MMFineReason-sft 1.77M) don't map the whole split.
    _max = os.environ.get("MAX_SAMPLES")
    if _max is not None:
        ds = ds.select(range(min(int(_max), len(ds))))
    # writer_batch_size kept small as a second guard against the 2GB Arrow offset
    # overflow (primary guard is the _cap_image call in _fmt): 100 capped images
    # per shard is comfortably under the int32 offset limit even for large images.
    return ds.map(_fmt, remove_columns=ds.column_names, writer_batch_size=100)


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
    if dataset_name in _SPECS:
        full_train = _load_spec_dataset(dataset_name)
    elif dataset_name in (CLEVR_COUNTING_DATASET, GEOQA_DATASET):
        raw = hf_load_dataset(dataset_name)
        train_split = raw["train"]
        columns = set(train_split.column_names)
        # R1-V datasets standardize on `problem` + `solution` + `image`.
        if not {"image", "problem", "solution"} <= columns:
            raise ValueError(
                f"Dataset '{dataset_name}' must have 'problem'/'solution'/'image'. "
                f"Found columns: {columns}"
            )

        def _format(example):
            # CLEVR/GEOQA store solution as '<answer> X </answer>'. Strip the
            # wrapper so `solution` is the bare gold (e.g. '3', '145°') — matches
            # what reward_correctness extracts from completions.
            raw_sol = str(example["solution"])
            stripped = extract_answer_tag(raw_sol)
            return {
                "prompt": _make_prompt(example["problem"]),
                "image": example["image"],
                "solution": stripped if stripped is not None else raw_sol.strip(),
            }

        full_train = train_split.map(_format, remove_columns=train_split.column_names)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: "
            f"GEOQA/CLEVR + {sorted(_SPECS)}"
        )

    # Decode `image` → PIL (no-op if already) and cap long side. Shared by all
    # sources so Qwen's dynamic-resolution tiling can't exceed max_model_len.
    if not isinstance(full_train.features["image"], HFImage):
        full_train = full_train.cast_column("image", HFImage())
    full_train = full_train.map(_convert_to_rgb, writer_batch_size=100)

    eval_path = os.environ.get("MLLM_EVAL_PATH")
    if eval_path is not None:
        # A real fixed eval set is provided → train on ALL of full_train (no
        # 150-holdout carve, which also avoids train_test_split crashing when
        # MAX_SAMPLES truncates full_train below _VALIDATION_SIZE).
        train_dataset = full_train
        eval_dataset = _load_local_eval_jsonl(eval_path, os.environ.get("MLLM_EVAL_IMAGE_DIR"))
    else:
        # No eval set: carve a 150 holdout from train (seed 42).
        split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
        train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))

    return train_dataset, eval_dataset


__all__ = [
    "CLEVR_COUNTING_DATASET",
    "GEOQA_DATASET",
    "load_dataset",
]
