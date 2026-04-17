"""Re-export majority-vote utilities from un-grpo-maj for co-learning GRPO."""

import importlib.util
from pathlib import Path

_slt_file = Path(__file__).resolve().parent.parent / "un-grpo-maj" / "self_label_trainer.py"
_spec = importlib.util.spec_from_file_location("_un_grpo_maj_self_label_trainer", str(_slt_file))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_UNLABELED_SENTINEL = _mod._UNLABELED_SENTINEL
_extract_and_normalize = _mod._extract_and_normalize
_majority_vote = _mod._majority_vote
extract_boxed_answer = _mod.extract_boxed_answer
normalize_answer = _mod.normalize_answer

__all__ = [
    "extract_boxed_answer",
    "normalize_answer",
    "_extract_and_normalize",
    "_majority_vote",
    "_UNLABELED_SENTINEL",
]
