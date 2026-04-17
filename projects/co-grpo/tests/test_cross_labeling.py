"""Unit tests for co-learning cross-labeling logic."""

import sys
from pathlib import Path

# Add project dirs to path
_co_grpo_dir = str(Path(__file__).resolve().parent.parent)
_un_grpo_maj_dir = str(Path(__file__).resolve().parent.parent.parent / "un-grpo-maj")
for d in [_co_grpo_dir, _un_grpo_maj_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _majority_vote,
    extract_boxed_answer,
    normalize_answer,
)


class TestExtractBoxedAnswer:
    def test_simple(self):
        assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        assert extract_boxed_answer("no boxed answer here") is None

    def test_empty(self):
        assert extract_boxed_answer(r"\boxed{}") == ""


class TestNormalizeAnswer:
    def test_strips_and_lowercases(self):
        assert normalize_answer("  Hello World  ") == "hello world"

    def test_none(self):
        assert normalize_answer(None) is None


class TestMajorityVote:
    def test_unanimous(self):
        label, freq = _majority_vote(["42", "42", "42", "42"], threshold=0.0)
        assert label == "42"
        assert freq == 1.0

    def test_plurality(self):
        label, freq = _majority_vote(["42", "42", "7", "7", "42"], threshold=0.0)
        assert label == "42"
        assert abs(freq - 3 / 5) < 1e-6

    def test_threshold_reject(self):
        # 2 out of 4 = 0.5, threshold 0.6 rejects
        label, freq = _majority_vote(["42", "42", "7", "7"], threshold=0.6)
        assert label is None
        assert abs(freq - 0.5) < 1e-6

    def test_all_none(self):
        label, freq = _majority_vote([None, None, None], threshold=0.0)
        assert label is None
        assert freq == 0.0

    def test_mixed_with_nones(self):
        label, freq = _majority_vote([None, "42", None, "42"], threshold=0.0)
        assert label == "42"
        assert freq == 1.0  # 2/2 parseable, both "42"


class TestCrossLabelingLogic:
    """Test the cross-labeling protocol on canned completions."""

    def test_basic_cross_labeling(self):
        """Two groups (G=4): A says '42' for both, B says '7' for both.
        Cross-labeling: A gets '7' (B's label), B gets '42' (A's label)."""
        G = 4
        # Model A: group 0 says "42", group 1 says "42"
        completions_a = [r"\boxed{42}"] * 4 + [r"\boxed{42}"] * 4
        # Model B: group 0 says "7", group 1 says "7"
        completions_b = [r"\boxed{7}"] * 4 + [r"\boxed{7}"] * 4

        answers_a = [_extract_and_normalize(c) for c in completions_a]
        answers_b = [_extract_and_normalize(c) for c in completions_b]
        num_groups = len(completions_a) // G

        pseudo_labels_a = []  # A's own labels (given to B)
        pseudo_labels_b = []  # B's own labels (given to A)
        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label_a, _ = _majority_vote(answers_a[lo:hi], 0.0)
            label_b, _ = _majority_vote(answers_b[lo:hi], 0.0)
            pseudo_labels_a.append(label_a if label_a is not None else _UNLABELED_SENTINEL)
            pseudo_labels_b.append(label_b if label_b is not None else _UNLABELED_SENTINEL)

        # A's own labels
        assert pseudo_labels_a == ["42", "42"]
        # B's own labels
        assert pseudo_labels_b == ["7", "7"]

        # Cross-labeling: A gets B's labels, B gets A's labels
        labels_for_a = pseudo_labels_b  # ["7", "7"]
        labels_for_b = pseudo_labels_a  # ["42", "42"]

        # Now check rewards:
        # A's completions say "42", but A is scored against B's label "7" → reward 0
        for comp in completions_a:
            pred = normalize_answer(extract_boxed_answer(comp))
            for label in labels_for_a:
                assert pred != label  # "42" != "7"

        # B's completions say "7", scored against A's label "42" → reward 0
        for comp in completions_b:
            pred = normalize_answer(extract_boxed_answer(comp))
            for label in labels_for_b:
                assert pred != label  # "7" != "42"

    def test_agreeing_models(self):
        """When both models agree, cross-labeling gives reward 1.0."""
        G = 4
        # Both say "42"
        completions_a = [r"\boxed{42}"] * 4
        completions_b = [r"\boxed{42}"] * 4

        answers_a = [_extract_and_normalize(c) for c in completions_a]
        answers_b = [_extract_and_normalize(c) for c in completions_b]

        label_a, _ = _majority_vote(answers_a, 0.0)
        label_b, _ = _majority_vote(answers_b, 0.0)

        # Cross-label: A gets B's "42", B gets A's "42"
        assert label_a == "42"
        assert label_b == "42"

        # A's completions say "42", scored against B's "42" → reward 1.0
        for comp in completions_a:
            pred = normalize_answer(extract_boxed_answer(comp))
            assert pred == label_b

    def test_mixed_group(self):
        """A has mixed answers, B is unanimous. Cross-labeling: some of A's rollouts match."""
        G = 4
        completions_a = [r"\boxed{42}", r"\boxed{42}", r"\boxed{42}", r"\boxed{7}"]
        completions_b = [r"\boxed{42}"] * 4

        answers_a = [_extract_and_normalize(c) for c in completions_a]
        answers_b = [_extract_and_normalize(c) for c in completions_b]

        label_a, _ = _majority_vote(answers_a, 0.0)  # "42" (3/4)
        label_b, _ = _majority_vote(answers_b, 0.0)  # "42" (4/4)

        assert label_a == "42"
        assert label_b == "42"

        # Cross-label: A scored against B's "42"
        # A's first 3 rollouts say "42" → reward 1.0; last says "7" → reward 0.0
        preds_a = [normalize_answer(extract_boxed_answer(c)) for c in completions_a]
        rewards_a = [1.0 if pred == label_b else 0.0 for pred in preds_a]
        assert rewards_a == [1.0, 1.0, 1.0, 0.0]

    def test_unlabeled_sentinel(self):
        """If B has no parseable answers, A gets the sentinel → all rewards 0."""
        G = 4
        completions_a = [r"\boxed{42}"] * 4
        completions_b = ["no answer here"] * 4  # nothing parseable

        answers_b = [_extract_and_normalize(c) for c in completions_b]
        label_b, _ = _majority_vote(answers_b, 0.0)
        assert label_b is None

        pseudo_b = _UNLABELED_SENTINEL
        # A scored against sentinel → "42" != sentinel → reward 0
        pred = normalize_answer(extract_boxed_answer(completions_a[0]))
        assert pred != pseudo_b
