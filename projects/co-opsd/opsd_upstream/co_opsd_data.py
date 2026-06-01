"""Data layer for co-OPSD (two-model on-policy co-distillation).

co-OPSD trains two models that distill into each other. Each model is driven by
its **own** data stream, described by a `ModelDataSpec`. The four supported data
regimes are all expressed through the same three fields — no special-casing:

| regime                       | model1 spec              | model2 spec              |
|-------------------------------|--------------------------|--------------------------|
| same data                     | (D, "train", 42)        | (D, "train", 42)         |
| same data, different shuffle   | (D, "train", 42)        | (D, "train", 7)          |
| different data                 | (D1, "train", 42)       | (D2, "train", 42)        |
| same data, different subset    | (D, "train[:50%]", 42)  | (D, "train[50%:]", 42)   |

`PairedDataset` zips the two streams so a single HF `Trainer` dataloader yields,
per step, one example for each model. `CoSelfDistillationDataCollator` then
builds, with each model's own tokenizer:

  - the **student** prompt (problem only) the model generates its trajectory from
  - the **teacher** prompt the *other* model scores that trajectory with
    (problem + ground-truth solution iff `teacher_sees_gt_answer`)
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset


# Instruction shared by student and teacher prompts (matches OPSD's data_collator.py).
_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Transition prompt appended after the reference solution when the teacher sees the
# ground-truth answer. Word-for-word from OPSD's SelfDistillationDataCollator.
_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


@dataclass
class ModelDataSpec:
    """Describes the data stream for one model.

    Args:
        dataset (`str`):
            HF Hub dataset name or local path.
        split (`str`, *optional*, defaults to `"train"`):
            HF split string. Subsets are expressed via slicing, e.g. `"train[:50%]"`.
        shuffle_seed (`int`, *optional*, defaults to `42`):
            Per-model shuffle seed. Two models sharing a dataset but with different
            seeds realises the "same data, different shuffle" regime.
        problem_column (`str`, *optional*, defaults to `"problem"`):
            Column holding the problem statement.
        solution_column (`str`, *optional*, defaults to `"solution"`):
            Column holding the ground-truth solution.
    """

    dataset: str
    split: str = "train"
    shuffle_seed: int = 42
    problem_column: str = "problem"
    solution_column: str = "solution"


def _load_one(spec: ModelDataSpec):
    """Load one model's stream and normalise its columns to `problem` / `solution`."""
    ds = load_dataset(spec.dataset, split=spec.split)
    ds = ds.shuffle(seed=spec.shuffle_seed)
    ds = ds.map(
        lambda ex: {"problem": ex[spec.problem_column], "solution": ex[spec.solution_column]},
        remove_columns=ds.column_names,
    )
    return ds


class PairedDataset(torch.utils.data.Dataset):
    """Zips the two models' streams; length is the shorter of the two."""

    def __init__(self, spec1: ModelDataSpec, spec2: ModelDataSpec):
        self.ds1 = _load_one(spec1)
        self.ds2 = _load_one(spec2)
        self.length = min(len(self.ds1), len(self.ds2))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"model1": self.ds1[i], "model2": self.ds2[i]}


def _student_prompt(tokenizer, problem: str) -> str:
    """Prompt the model generates its on-policy trajectory from (problem only)."""
    messages = [{"role": "user", "content": f"Problem: {problem}\n\n{_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _teacher_prompt(tokenizer, problem: str, solution: str, sees_gt_answer: bool) -> str:
    """Prompt the *other* model scores the trajectory with.

    With `sees_gt_answer=True` the ground-truth solution is embedded (OPSD's
    privileged teacher); with `False` it degrades to the plain problem prompt and
    the only signal comes from the two models' diversity.
    """
    if not sees_gt_answer:
        return _student_prompt(tokenizer, problem)
    user_message = (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{_TRANSITION_PROMPT}\n"
        f"{_INSTRUCTION}"
    )
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def _encode(tokenizer, texts: list[str], max_length: int) -> dict:
    """Tokenize a list of prompts, left-padded to the batch max.

    Left padding (set on the tokenizers in the collator `__init__`) makes every
    prompt end at the same offset, so a single `prompt_len` slices the trajectory
    region uniformly. Returns `input_ids`, `attention_mask`, and that length.
    """
    no_pad = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    batch_max = max(len(ids) for ids in no_pad["input_ids"])
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=batch_max,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "prompt_len": batch_max,
    }


class CoSelfDistillationDataCollator:
    """Builds student/teacher prompts for both models, each with its own tokenizer.

    Per batch, model1's examples yield model1's student prompt (`m1_student_*`,
    encoded with `tokenizer1`) and the teacher prompt model2 scores model1's
    trajectory with (`m2_teacher_*`, `tokenizer2`); model2's examples yield the
    symmetric `m2_student_*` / `m1_teacher_*` groups.
    """

    def __init__(self, tokenizer1, tokenizer2, max_length=2048, teacher_sees_gt_answer=True):
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.max_length = max_length
        self.teacher_sees_gt_answer = teacher_sees_gt_answer
        # Left padding: the student prompts are fed to `.generate()` (decoder-only
        # generation requires left padding) and, for RoPE models, a left-padded
        # forward pass is equivalent since RoPE positions are relative.
        self.tokenizer1.padding_side = "left"
        self.tokenizer2.padding_side = "left"

    def __call__(self, features):
        ex1 = [f["model1"] for f in features]
        ex2 = [f["model2"] for f in features]

        # model1 generates its trajectory; model2 scores it.
        m1_student = [_student_prompt(self.tokenizer1, e["problem"]) for e in ex1]
        m2_teacher = [
            _teacher_prompt(self.tokenizer2, e["problem"], e["solution"], self.teacher_sees_gt_answer)
            for e in ex1
        ]
        # model2 generates its trajectory; model1 scores it.
        m2_student = [_student_prompt(self.tokenizer2, e["problem"]) for e in ex2]
        m1_teacher = [
            _teacher_prompt(self.tokenizer1, e["problem"], e["solution"], self.teacher_sees_gt_answer)
            for e in ex2
        ]

        groups = {
            "m1_student": _encode(self.tokenizer1, m1_student, self.max_length),
            "m2_teacher": _encode(self.tokenizer2, m2_teacher, self.max_length),
            "m2_student": _encode(self.tokenizer2, m2_student, self.max_length),
            "m1_teacher": _encode(self.tokenizer1, m1_teacher, self.max_length),
        }

        result = {}
        for name, enc in groups.items():
            result[f"{name}_input_ids"] = enc["input_ids"]
            result[f"{name}_attention_mask"] = enc["attention_mask"]
            result[f"{name}_prompt_len"] = enc["prompt_len"]
        return result


def build_paired_dataset(spec1: ModelDataSpec, spec2: ModelDataSpec) -> PairedDataset:
    """Convenience constructor used by the training entrypoint."""
    return PairedDataset(spec1, spec2)
