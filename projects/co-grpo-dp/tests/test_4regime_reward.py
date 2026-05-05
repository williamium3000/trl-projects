"""Unit tests for `compute_4regime_reward` (confidence-gated 4-regime reward).

Pure CPU; no model or GPU needed. Verifies:
1. Each regime fires under correct conditions.
2. Boundary semantics: tau_high and tau_mid are inclusive (>=).
3. Frequencies use total N as denominator (not valid-only count).
4. None handling: my_answer is None -> unseen; all-None peer -> no_valid_peer.
5. Lambda scaling: negative reward magnitude = lambda_.
6. Determinism: same input always produces same output.
"""

import sys
import warnings
from pathlib import Path

import pytest

_co_grpo_dp_dir = str(Path(__file__).resolve().parent.parent)
if _co_grpo_dp_dir not in sys.path:
    sys.path.insert(0, _co_grpo_dp_dir)

warnings.filterwarnings("ignore", category=SyntaxWarning)

from co_label_utils import compute_4regime_reward  # noqa: E402


# Default starting hyperparameters used by N=8 experiments
TAU_HIGH = 5 / 8  # 0.625
TAU_MID = 2 / 8   # 0.25
LAMBDA = 0.5


# ---------------- regime: confident_positive ----------------

def test_confident_positive_clear_majority():
    # 6/8 = 0.75 >= 0.625 -> confident
    peer = ["a"] * 6 + ["b"] * 2
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 1.0
    assert regime == "confident_positive"


def test_confident_positive_unanimous():
    peer = ["a"] * 8
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 1.0
    assert regime == "confident_positive"


def test_confident_positive_at_tau_high_boundary():
    # 5/8 = 0.625 == tau_high -> inclusive, fires
    peer = ["a"] * 5 + ["b"] * 3
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 1.0
    assert regime == "confident_positive"


# ---------------- regime: uncertain_majority ----------------

def test_uncertain_majority_just_below_tau_high():
    # a:4 unique top, c_top=4/8=0.5 < 0.625 -> uncertain (no tie at top)
    peer = ["a"] * 4 + ["b"] * 3 + ["c"]
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "uncertain_majority"


def test_uncertain_majority_low_confidence():
    # plurality 3/8 = 0.375 < 0.625
    peer = ["a"] * 3 + ["b"] * 2 + ["c"] * 2 + ["d"] * 1
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "uncertain_majority"


# ---------------- regime: strong_runner_up ----------------

def test_strong_runner_up_at_tau_mid_boundary():
    # p_b = 2/8 = 0.25 == tau_mid -> inclusive, strong
    peer = ["a"] * 5 + ["b"] * 2 + ["c"] * 1
    r, regime = compute_4regime_reward("b", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "strong_runner_up"


def test_strong_runner_up_above_tau_mid():
    # p_b = 3/8 = 0.375 > 0.25
    peer = ["a"] * 4 + ["b"] * 3 + ["c"] * 1
    r, regime = compute_4regime_reward("b", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "strong_runner_up"


# ---------------- regime: weak ----------------

def test_weak_just_below_tau_mid():
    # p_c = 1/8 = 0.125 < 0.25
    peer = ["a"] * 5 + ["b"] * 2 + ["c"] * 1
    r, regime = compute_4regime_reward("c", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == -LAMBDA
    assert regime == "weak"


# ---------------- regime: unseen ----------------

def test_unseen_my_not_in_peer():
    peer = ["a"] * 8
    r, regime = compute_4regime_reward("zzz", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == -LAMBDA
    assert regime == "unseen"


def test_unseen_my_answer_is_none():
    # Extraction failed on B's rollout — treated as unseen.
    peer = ["a"] * 8
    r, regime = compute_4regime_reward(None, peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == -LAMBDA
    assert regime == "unseen"


# ---------------- regime: no_valid_peer ----------------

def test_no_valid_peer_all_none():
    peer = [None] * 8
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "no_valid_peer"


def test_no_valid_peer_empty_list():
    r, regime = compute_4regime_reward("a", [], TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "no_valid_peer"


# ---------------- regime: tied_top ----------------

def test_tied_top_all_unique():
    # 8-way tie at count=1 (every answer unique). Without short-circuit, the
    # arbitrary first-inserted answer would get reward 0 while all others got
    # -lambda — pure order-dependent noise. With the fix, all rollouts get 0.
    peer = ["a", "b", "c", "d", "e", "f", "g", "h"]
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_tied_top_two_way_at_tau_mid():
    # 2-way tie at top with count=2 each (== tau_mid). Conservative fix:
    # short-circuit on any top tie regardless of count magnitude.
    peer = ["a", "a", "b", "b", "c", "d", "e", "f"]
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_tied_top_three_way_above_tau_mid():
    # 3-way tie at count=3 each (3/8=0.375 > tau_mid). Still tied_top.
    peer = ["a", "a", "a", "b", "b", "b", "c", "c"]
    # Fixing one tie: a:3, b:3, c:2  -> a and b tie at top.
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_tied_top_my_not_in_peer():
    # All-unique peer + my_answer also unique. tied_top still fires (preempts
    # the my != top -> unseen path).
    peer = ["a", "b", "c", "d", "e", "f", "g", "h"]
    r, regime = compute_4regime_reward("zzz", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_tied_top_preempts_my_answer_none():
    # Tied-top check fires before the my_answer=None -> unseen check.
    # This is intentional: tied_top means peer signal is noise, so we shouldn't
    # penalize my=None against an unreliable signal.
    peer = ["a", "b", "c", "d", "e", "f", "g", "h"]
    r, regime = compute_4regime_reward(None, peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_no_tie_clear_winner_unaffected():
    # Sanity: clear unique top -> tied_top does NOT fire (no false positive).
    peer = ["a"] * 4 + ["b", "b", "c", "d"]  # a:4 unique top, b:2, c:1, d:1
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert regime == "uncertain_majority"  # 4/8=0.5 < 0.625
    assert r == 0.0
    r2, regime2 = compute_4regime_reward("b", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert regime2 == "strong_runner_up"  # 2/8 = tau_mid (inclusive)
    assert r2 == 0.0


def test_tied_top_with_partial_none_peer():
    # Ties counted over valid_peer only (None entries are not "candidates").
    # peer = [a, a, b, b, None, None, None, None]: valid={a:2, b:2} -> tied.
    peer = ["a", "a", "b", "b", None, None, None, None]
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "tied_top"


def test_single_valid_answer_is_not_tied():
    # peer = [a, None, None, None, None, None, None, None]: valid={a:1}, no tie.
    # c_top = 1/8 = 0.125. my=a -> uncertain_majority, my=other -> weak/unseen.
    peer = ["a"] + [None] * 7
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "uncertain_majority"


# ---------------- frequency uses total N (not valid count) ----------------

def test_partial_none_peer_lowers_c_top():
    # 4 valid all "a", 4 None. c_top = 4/8 = 0.5 < 0.625 -> uncertain (not confident)
    peer = ["a"] * 4 + [None] * 4
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 0.0
    assert regime == "uncertain_majority"


def test_partial_none_peer_can_still_be_confident():
    # 6 valid all "a", 2 None. c_top = 6/8 = 0.75 >= 0.625
    peer = ["a"] * 6 + [None] * 2
    r, regime = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert r == 1.0
    assert regime == "confident_positive"


# ---------------- lambda scaling ----------------

def test_lambda_one_yields_negative_one():
    peer = ["a"] * 8
    r, regime = compute_4regime_reward("zzz", peer, TAU_HIGH, TAU_MID, 1.0)
    assert r == -1.0
    assert regime == "unseen"


def test_lambda_zero_yields_zero_negative():
    # lambda=0 effectively disables the negative penalty.
    peer = ["a"] * 5 + ["b"] * 2 + ["c"] * 1
    r, regime = compute_4regime_reward("c", peer, TAU_HIGH, TAU_MID, 0.0)
    assert r == 0.0  # weak regime fires but penalty is zero
    assert regime == "weak"


# ---------------- extreme thresholds ----------------

def test_tau_high_one_requires_unanimous():
    # tau_high=1.0 — only c_top == 1.0 fires confident_positive
    peer = ["a"] * 7 + ["b"] * 1
    r, regime = compute_4regime_reward("a", peer, 1.0, TAU_MID, LAMBDA)
    assert regime == "uncertain_majority"
    peer_unanimous = ["a"] * 8
    r, regime = compute_4regime_reward("a", peer_unanimous, 1.0, TAU_MID, LAMBDA)
    assert regime == "confident_positive"


def test_tau_mid_zero_makes_any_seen_strong():
    # tau_mid=0 — any p_my > 0 maps to strong_runner_up (since p_my >= 0)
    # Note p_my == 0 still routes to unseen via the explicit check.
    peer = ["a"] * 5 + ["b"] * 2 + ["c"] * 1
    r, regime = compute_4regime_reward("c", peer, TAU_HIGH, 0.0, LAMBDA)
    assert regime == "strong_runner_up"


# ---------------- determinism ----------------

def test_determinism_repeat_calls():
    peer = ["a", "b", "a", "c", "a", "b", "a", "d"]
    r1, regime1 = compute_4regime_reward("b", peer, TAU_HIGH, TAU_MID, LAMBDA)
    r2, regime2 = compute_4regime_reward("b", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert (r1, regime1) == (r2, regime2)


def test_determinism_tied_top_short_circuit():
    # Tied-top inputs now short-circuit to (0.0, "tied_top") deterministically,
    # avoiding the order-dependent Counter.most_common tie-breaking that the
    # previous version exposed.
    peer = ["a"] * 4 + ["b"] * 4
    r1, regime1 = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    r2, regime2 = compute_4regime_reward("a", peer, TAU_HIGH, TAU_MID, LAMBDA)
    assert (r1, regime1) == (r2, regime2) == (0.0, "tied_top")
