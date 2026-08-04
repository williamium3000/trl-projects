"""Edge-case checks for the --peer_label_mode additions.

verify_modes.sh covers the happy path. This covers the way the new code can kill
a run rather than degrade a number: the metrics block executes on every training
step, so an exception there takes down a three-day job at step 1.

The specific regression introduced: `gt_answers` calls normalize_answer() for
every prompt, where the original only called it inside the branch that already
had a pseudo-label. A row with a null or unparseable solution was previously
never passed to it.
"""

import sys
import traceback

import co_grpo_dp_trainer as T
import train_co_grpo_dp as TR
from co_label_utils import _UNLABELED_SENTINEL as S
from co_label_utils import normalize_answer

SEP = T.CANDIDATE_SEP
fails = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        fails.append(name)


print("\n=== normalize_answer on null / junk ===")


def norm():
    assert normalize_answer(None) is None, "None did not pass through"
    for bad in ["", "   ", r"\boxed{}", "not an answer", "$$", r"\frac{"]:
        normalize_answer(bad)  # must not raise


check("normalize_answer(None) and malformed input", norm)


print("\n=== metrics arithmetic, same expressions as the trainer ===")


def metrics_edges():
    cases = [
        ("empty batch", 0, [], {"B": [], "C": []}),
        ("all unlabeled", 2, [S, S], {"B": [S, S], "C": [S, S]}),
        ("unanimous", 2, ["7", "9"], {"B": ["7", "9"], "C": ["7", "9"]}),
        ("mixed", 3, ["7", S, "5"], {"B": ["9", "7", "5"], "C": [S, "7", "4"]}),
    ]
    for name, ng, mine, peers in cases:
        pl = list(peers)
        all_p = [g for g in range(ng) if all(peers[p][g] != S for p in pl)]
        full = [g for g in all_p if mine[g] != S]
        una = sum(1 for g in full if all(peers[p][g] == mine[g] for p in pl))
        r_una = una / len(full) if full else 0.0

        sup = [T._peer_candidate_set([peers[p][g] for p in pl]) for g in range(ng)]
        sizes = [len(x.split(SEP)) for x in sup if x != S]
        r_size = sum(sizes) / len(sizes) if sizes else 0.0

        # gt None everywhere is the worst case for the new contains-truth loop
        for gts in ([None] * ng, ["7"] * ng):
            hits = seen = 0
            for g in range(ng):
                if sup[g] == S or gts[g] is None:
                    continue
                seen += 1
                if gts[g] in sup[g].split(SEP):
                    hits += 1
            _ = hits / seen if seen else 0.0
        assert 0.0 <= r_una <= 1.0 and r_size >= 0.0, name
        print(f"        {name}: unanimous={r_una:.2f} cand_size={r_size:.2f}")


check("empty / all-sentinel / unanimous / mixed", metrics_edges)


print("\n=== sentinel packed alongside a real candidate ===")


def sent():
    c = lambda t: [{"role": "assistant", "content": t}]
    assert TR.reward_correctness(completions=[c(r"\boxed{7}")], solution=[S + SEP + "9"]) == [0.0]
    assert TR.reward_correctness(completions=[c(r"\boxed{9}")], solution=[S + SEP + "9"]) == [1.0]


check("sentinel never scores, real candidate still does", sent)


print("\n=== self_plus_peers degenerate case is visible, not silent ===")


def degenerate():
    # Both peers unlabeled: the vote has only my own answer, so supervision is
    # pure self-labeling with no cross-model signal at all. Measured
    # labeled_fraction was 1.0 so this should never fire, but if it does the
    # run is quietly TTRL and supervision_contains_truth is how we would see it.
    assert T._self_plus_peer_vote("7", [S, S]) == "7"
    assert T._peer_candidate_set([S, S]) == S


check("documented degenerate behaviour holds", degenerate)


print("\n" + "=" * 54)
if fails:
    print("VERIFY2 FAILED:", ", ".join(fails))
    sys.exit(1)
print("VERIFY2 OK")
