# N=3 co-grpo-dp extension (2026-05-22)

> TODO §5.3.1 — full mutual N=3 cross-family co-learning.

## What changed

3 files in `projects/co-grpo-dp/`:

| file | change |
|---|---|
| `rendezvous.py` | + `exchange_n_way(mode, counter, payload) → dict[peer→payload]`. Per-(src,dst) directed file layout so each consumer deletes only its own incoming file (no inter-consumer race). Old 2-way `exchange()` kept untouched. |
| `co_grpo_dp_trainer.py` | + helper `_peer_majority_vote(peer_labels) → label`. `__init__` sources `self.peers` from `rendezvous.peers`. `_calculate_rewards` branches on `self.n_way`: 2-way path is byte-identical to before; N-way (N≥3) path does MV over peer pseudos with **strict-tie → UNLABELED**. New per-peer metrics + `co_labeling/supervision_fraction` + `co_labeling/peer_tie_rate`. |
| `train_co_grpo_dp.py` | + `--peers "B,C"` arg. `--group` accepts any single uppercase letter. Seed offset generalizes to `ord(group) - ord('A')` so A=0, B=+1, C=+2. Rendezvous constructed with `peers=peer_groups`. |

Plus new launch script + this doc. No breaking change for existing 2-way scripts (`exchange()`, `.peer` attr, A/B group names all preserved).

## Algorithm (N=3 specifically)

Each generation step, for each prompt group `g` (= 12 rollouts per model per prompt):

```
for each group ∈ {A, B, C} in parallel:
    1. K=12 SC vote internal      → my_pseudo[g]   (or UNLABELED if internal SC fails)
    2. exchange_n_way:            send my_pseudo to (B, C);
                                  receive peer_pseudos = {B: ..., C: ...}
    3. supervision[g] = MV({peer_pseudos[B][g], peer_pseudos[C][g]})
         - if B[g] == C[g]   → that value wins (strict majority)
         - if B[g] != C[g]   → UNLABELED (strict tie, discard per §5.3)
         - if either is UNLABELED → ignore it, take the other (1-of-1 = unanimous)
         - if both UNLABELED → UNLABELED
    4. inject supervision[g] into inputs[i]["solution"] for the 12 local rollouts
    5. delegate to GRPOTrainer parent → reward = grade_answer(my_rollout, supervision)
       (UNLABELED → reward = 0 for every rollout in that group → no learning signal that step)
```

## Backward compatibility

- All existing 2-way (A↔B) scripts keep running with the legacy
  `Rendezvous(rendezvous_dir, my_group_name)` signature (peers=None defaults to
  the legacy pair). Their wandb dashboards see the same metric keys
  (`co_labeling/peer_agreement`, `co_labeling/labeled_fraction_peer`,
  `co_labeling/both_labeled_fraction`).
- N-way (N≥3) wandb dashboards see the **per-peer** keys instead:
  `co_labeling/peer_agreement/<peer>` `co_labeling/labeled_fraction_peer/<peer>`
  plus **N-way only**:
  - `co_labeling/supervision_fraction` — fraction of prompts where peers gave
    a non-UNLABELED majority (i.e. this group got useful supervision)
  - `co_labeling/peer_tie_rate` — within prompts where ALL N-1 peers were labeled,
    what fraction came back as ties → UNLABELED (the "平票丢弃" rate)

## 8-GPU allocation

8 cards / 3 groups doesn't divide evenly. Three options:

| split | wasted | symmetric? | grad_accum/group | step time vs 2-way |
|---|---|---|---|---|
| **2+2+2 (default)** | 2 cards | ✅ | 768 | ~2× slower |
| 3+3+2 | 0 cards | ❌ A/B faster than C | 512 / 512 / 768 | ~1.3× / ~2× |
| 4+4+4 across 2 pod | 0 cards | ✅ | 384 | matches 2-way speed |

Default script uses 2+2+2 single-pod. To run 4+4+4 multi-pod, change CUDA_VISIBLE_DEVICES per group and update main_process_port + rendezvous_dir to live on a shared NFS path.

## Tie-discard semantics (paper section)

Two ways to "vote among peers":
- **Plurality**: pick the most-common; on a tie, pick arbitrarily (or by lexicographic rule)
- **Strict majority**: pick the most-common only if strictly greater than all others; on a tie, **discard** (this is what TODO §5.3 prescribes)

We use strict majority. For N=3:
- 2/2 peers agree → label wins (this is also strict majority of 2 valid votes)
- 2 peers disagree (1 vs 1) → tie → DISCARD
- 1 of 2 peers unlabeled → the other peer's label wins (degenerate to 2-way co-learn for that step)
- both unlabeled → DISCARD

For N=4+, strict-majority means: a clear plurality with no co-leader at the top count.

The tie-discard rate is logged as `co_labeling/peer_tie_rate` — useful as a paper appendix to show how often the N=3 protocol falls back to UNLABELED.

## Smoke test (CPU, no GPU needed)

```python
# Rendezvous N=3 (uses /tmp + threads, < 1s)
python -c "
import sys, tempfile, threading
sys.path.insert(0, 'projects/co-grpo-dp')
from rendezvous import Rendezvous

with tempfile.TemporaryDirectory() as tmp:
    results = {}
    def run(name, peers, payload):
        r = Rendezvous(tmp, name, poll_interval=0.01, peers=peers)
        results[name] = r.exchange_n_way('train', 0, payload)
    ts = [threading.Thread(target=run, args=(n, [p for p in 'ABC' if p != n], [f'{n}-says']))
          for n in 'ABC']
    for t in ts: t.start()
    for t in ts: t.join()
    print(results)
"
# → {'A': {'B': ['B-says'], 'C': ['C-says']}, 'B': ..., 'C': ...}
```

```python
# MV edge cases
python -c "
import sys; sys.path.insert(0, 'projects/co-grpo-dp')
from co_grpo_dp_trainer import _peer_majority_vote
from co_label_utils import _UNLABELED_SENTINEL as U
assert _peer_majority_vote(['7','7']) == '7'           # both-agree
assert _peer_majority_vote(['7','11']) == U            # tie → discard
assert _peer_majority_vote(['7','7','11']) == '7'      # 2-of-3
assert _peer_majority_vote(['7','11','13']) == U       # 3-way tie
assert _peer_majority_vote(['7', U]) == '7'            # 1 unlabeled, take other
"
```

Both passed 2026-05-22.

## Launch

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
```

Runs 3 accelerate worlds on CUDA 0,1 / 2,3 / 4,5. Each writes ckpt to
`projects/work_dirs/co-grpo-dp/cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b__math345_full_lr3e-6_e2_<TS>/group_{A,B,C}/`.

Best-by-val select per group:

```bash
RUN_DIR=projects/work_dirs/co-grpo-dp/cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b__math345_full_lr3e-6_e2_<TS>
for grp in A B C; do
    python projects/eval/select_best_ckpt.py \
        --work_dir "$RUN_DIR/group_$grp" --top_k 5
done
```

## Test-time SC ensemble counterpart

N=3 test-time (no training, just inference) already works via the existing
`run_test_time_ensemble.sh` — pool size = K × N = 12 × 3 = **36** samples per
problem majority-voted across all 3 models. See TODO §4.7.4.

```bash
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" \
    --k 12 --gpu 0
# → 36-sample MV ensemble row in baselines.csv
```

This is the **untrained** ablation we compare the trained N=3 co-learn against
(paper §4.4.6 — the "test-time ensemble vs heter co-learn" comparison).

## Known risks

- **Rendezvous deadlock if one group crashes silently**: the surviving groups
  will TimeoutError after `timeout=3600s` (1 hour). Real crash → watch
  `wait -n` exits non-zero → cleanup() kills the rest. Don't lower timeout
  too aggressively; first save at step 10 already takes ~10-15 min on N=3.
- **2+2+2 is slow**: grad_accum=768 means ~768 forward-backwards per
  optimizer step. Step time on Blackwell ~6-8 min/step (vs ~3 min for N=2
  4+4 split). 117 steps × 2 epoch × 7 min ≈ **27 hours/run**.
- **3 vLLM colocate processes** on same node may contend for KV cache pages
  even at gpu_mem=0.40-0.45 per group. If OOM, lower one of them or move
  to multi-pod 4+4+4.
- **Wandb run name conflicts**: each group's wandb run has `_groupX` suffix;
  if you re-run with same TS (impossible — TS is `date +%s`), they'd collide.
