"""Re-export dataset loaders from un-grpo-maj."""

import importlib.util
from pathlib import Path

_dataset_file = Path(__file__).resolve().parent.parent / "un-grpo-maj" / "dataset.py"
_spec = importlib.util.spec_from_file_location("_un_grpo_maj_dataset", str(_dataset_file))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DAPO_DATASET = _mod.DAPO_DATASET
OPSD_DATASET = _mod.OPSD_DATASET
MATH_LEVEL345_DATASET = _mod.MATH_LEVEL345_DATASET
MATH_LEVEL12345_DATASET = _mod.MATH_LEVEL12345_DATASET
load_dataset = _mod.load_dataset
