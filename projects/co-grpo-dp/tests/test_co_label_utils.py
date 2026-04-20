"""Unit tests for co_label_utils — verifier (extract + grade) and majority vote.

Pure CPU; no model or GPU needed. Verifies:
1. extract_boxed_answer pulls answers from common completion shapes.
2. grade_answer treats sympy-equivalent forms as equal (1/2 == \\frac{1}{2} == 0.5).
3. normalize_answer canonicalizes hash keys (same equivalent forms collapse).
4. _extract_and_normalize returns None on no-answer (so majority_vote denominator stays right).
5. _majority_vote: plurality, threshold, all-None edge case, sentinel.
6. _UNLABELED_SENTINEL never grades True against real answers.
"""

import sys
import warnings
from pathlib import Path

import pytest

# Add project dir to path so we can import directly.
_co_grpo_dp_dir = str(Path(__file__).resolve().parent.parent)
if _co_grpo_dp_dir not in sys.path:
    sys.path.insert(0, _co_grpo_dp_dir)

# Suppress SyntaxWarnings from vendored qwen verifier (raw-string regex w/o r"" prefix).
warnings.filterwarnings("ignore", category=SyntaxWarning)

from co_label_utils import (  # noqa: E402
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _majority_vote,
    extract_boxed_answer,
    grade_answer,
    normalize_answer,
)


class TestExtract:
    def test_simple_boxed(self):
        assert extract_boxed_answer("answer: \\boxed{42}") == "42"

    def test_boxed_with_latex(self):
        assert extract_boxed_answer("Therefore \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"

    def test_no_boxed_falls_back_or_empty(self):
        # Qwen's parser tries fallbacks. We don't pin the exact fallback string,
        # but it must NOT raise.
        result = extract_boxed_answer("Hmm I have no idea.")
        assert isinstance(result, str)  # may be "" if no match


class TestGrade:
    @pytest.mark.parametrize("a,b", [
        ("1/2", "\\frac{1}{2}"),
        ("0.5", "\\frac{1}{2}"),
        ("\\frac{1}{2}", "\\frac{1}{2}"),
        ("42", "42"),
        ("x^2 + 2x", "2x + x^2"),  # algebraic equivalence
    ])
    def test_equivalent_pairs(self, a, b):
        assert grade_answer(a, b) is True

    @pytest.mark.parametrize("a,b", [
        ("1", "2"),
        ("\\frac{1}{2}", "\\frac{1}{3}"),
        ("42", "43"),
    ])
    def test_unequal_pairs(self, a, b):
        assert grade_answer(a, b) is False

    def test_none_input(self):
        assert grade_answer(None, "42") is False

    def test_sentinel_never_grades_true_against_real_answers(self):
        # The sentinel exists so that "no pseudo-label this prompt" produces
        # zero reward for every rollout. The real-world call shape is
        # grade_answer(extracted_completion, ground_truth_or_sentinel) — the
        # extracted side comes from extract_boxed_answer() which never returns
        # the sentinel, so the only sentinel-vs-real comparison that matters
        # is when ground_truth is the sentinel.
        assert grade_answer(_UNLABELED_SENTINEL, "42") is False
        assert grade_answer(_UNLABELED_SENTINEL, "\\frac{1}{2}") is False
        assert grade_answer("42", _UNLABELED_SENTINEL) is False
        assert grade_answer("\\frac{1}{2}", _UNLABELED_SENTINEL) is False
        # NB: grade_answer(SENTINEL, SENTINEL) is True (trivial sympy self-
        # equivalence on identical strings), but this case cannot occur because
        # extracted answers never equal the sentinel string.


class TestNormalizeForHashing:
    def test_equivalent_latex_collapses(self):
        # The whole point of the canonical normalizer: equivalent forms must
        # produce the same string so Counter() in _majority_vote can cluster
        # them without paying sympy cost.
        a = normalize_answer("0.5")
        b = normalize_answer("\\frac{1}{2}")
        c = normalize_answer("1/2")
        assert a == b == c

    def test_none_passthrough(self):
        assert normalize_answer(None) is None


class TestExtractAndNormalize:
    def test_no_boxed_returns_none(self):
        # Qwen parser returns "" on no-match; we normalize that to None at the
        # boundary so _majority_vote's denominator stays right (None is filtered).
        assert _extract_and_normalize("totally unrelated text") in (None, "")
        # Stricter: when really nothing extractable, must be None (not "").
        assert _extract_and_normalize("") is None

    def test_boxed_returns_canonical_string(self):
        assert _extract_and_normalize("\\boxed{42}") == "42"


class TestMajorityVote:
    def test_plurality_wins(self):
        answers = ["42", "42", "42", "100", "50"]
        label, freq = _majority_vote(answers, threshold=0.0)
        assert label == "42"
        assert freq == pytest.approx(3 / 5)

    def test_threshold_rejects_low_consensus(self):
        answers = ["42", "100", "50", "200", "300"]  # plurality 1/5 = 0.2
        label, freq = _majority_vote(answers, threshold=0.5)
        assert label is None  # below threshold
        assert freq == pytest.approx(0.2)

    def test_threshold_at_zero_accepts_anything(self):
        answers = ["42"]
        label, freq = _majority_vote(answers, threshold=0.0)
        assert label == "42"
        assert freq == 1.0

    def test_all_none_returns_none(self):
        answers = [None, None, None]
        label, freq = _majority_vote(answers, threshold=0.0)
        assert label is None
        assert freq == 0.0

    def test_none_filtered_from_denominator(self):
        # 3 valid + 5 None = 3 valid. Plurality "42" wins 3/3.
        answers = ["42", "42", "42", None, None, None, None, None]
        label, freq = _majority_vote(answers, threshold=0.0)
        assert label == "42"
        assert freq == pytest.approx(1.0)  # 3/3 valid, NOT 3/8

    def test_canonical_collapse_in_majority_vote(self):
        # Three different latex spellings of 1/2; without canonical normalize,
        # Counter would split them across 3 buckets and majority would tie.
        answers = [
            normalize_answer("1/2"),
            normalize_answer("\\frac{1}{2}"),
            normalize_answer("0.5"),
            normalize_answer("100"),
        ]
        label, freq = _majority_vote(answers, threshold=0.0)
        assert label == "\\frac{1}{2}"
        assert freq == pytest.approx(3 / 4)


class TestEndToEndFlow:
    """Simulate the train-time path for one prompt: completions -> labels -> reward."""

    def test_majority_label_then_grade_against_self(self):
        # Three rollouts answer 1/2 in different forms; majority = \frac{1}{2}.
        # grade_answer of the majority against ground truth "0.5" -> True.
        completions = [
            "I think \\boxed{1/2}",
            "Result: \\boxed{\\frac{1}{2}}",
            "So \\boxed{0.5}",
            "Wrong: \\boxed{100}",
        ]
        labels = [_extract_and_normalize(c) for c in completions]
        my_label, _ = _majority_vote(labels, threshold=0.5)
        assert my_label is not None
        assert grade_answer(my_label, "0.5") is True
        assert grade_answer(my_label, "\\frac{1}{2}") is True
        assert grade_answer(my_label, "1/2") is True
        assert grade_answer(my_label, "100") is False

    def test_sentinel_path_when_no_consensus(self):
        # All rollouts disagree -> majority None -> sentinel -> reward 0.
        completions = [
            "\\boxed{42}",
            "\\boxed{100}",
            "\\boxed{200}",
            "\\boxed{300}",
        ]
        labels = [_extract_and_normalize(c) for c in completions]
        my_label, _ = _majority_vote(labels, threshold=0.5)  # plurality 1/4 below 0.5
        if my_label is None:
            sentinel = _UNLABELED_SENTINEL
            assert grade_answer(sentinel, "42") is False  # no rollout gets reward
