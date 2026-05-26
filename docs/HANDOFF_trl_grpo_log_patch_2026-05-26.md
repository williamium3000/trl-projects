# Handoff: trl GRPOTrainer.log() 1-line patch — required for BestKeeperCallback to actually work

**Date**: 2026-05-26
**Files changed**: `trl/trainer/grpo_trainer.py` (1 line)
**TL;DR**: Your `BestKeeperCallback` design is correct, but a latent bug in
trl's GRPO trainer would have caused it to silently fail (never create
`best_model/`). One-line fix in `trl/trainer/grpo_trainer.py:2786` makes it
work. **Do not revert this line.**

---

## What was broken (subtle, silent failure)

`BestKeeperCallback.on_evaluate(args, state, control, metrics=None, **kw)`
reads `metrics["eval_reward"]` to decide whether to update best. But the
`metrics` dict it receives — handed off by HF Trainer's
`callback_handler.on_evaluate(..., metrics=metrics)` — is the **return
value of `Trainer.evaluation_loop()`**, which only contains the 4 base
metrics:

```python
{'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second'}
```

The reward / completion / sampling metrics (`eval_reward`, `eval_reward_std`,
`eval_completions/*`, etc.) are tracked separately in `self._metrics["eval"]`
and only added to a **fresh local dict** that's forwarded to `super().log()`
for printing / wandb:

```python
# trl/trainer/grpo_trainer.py — original (pre-patch)
def log(self, logs, start_time=None):
    mode = "train" if self.model.training else "eval"
    metrics = {}
    for key, val in self._metrics[mode].items():
        ...
    if mode == "eval":
        metrics = {f"eval_{key}": val for key, val in metrics.items()}

    logs = {**logs, **metrics}     # ← creates NEW dict, doesn't mutate input
    super().log(logs, start_time)  # ← prints / wandb sees the NEW dict
    self._metrics[mode].clear()
```

So:
- **Console / wandb** see the full dict (good — that's why we always saw
  `eval_reward` in train.log and wandb plots).
- **Caller's `metrics` dict** (the one HF Trainer passes onward to
  `callback_handler.on_evaluate`) is untouched — only has the 4 base keys.
- `BestKeeperCallback.on_evaluate(metrics=...)` receives the 4-key dict →
  `metrics.get("eval_reward")` returns `None` → callback returns early →
  `best_model/` **never gets created**.

Same root cause also breaks any naive use of
`metric_for_best_model=eval_reward + load_best_model_at_end=true` —
`Trainer._determine_best_metric` `KeyError`s on the same missing key.
(That path is independent of our callback; we hit it during smoke testing
when we briefly tried the HF-native load_best machinery.)

## How we verified

Without the patch, `best_model/` was always absent on disk after training
(grep `find work_dirs -name best_model` after commit 8246844d → empty).
With the patch, end-to-end smoke (`/tmp/grpo_partB_v12_*`) produced:

```
best_metric.json    {"step": 4, "metric": "eval_reward", "value": 0.0592}
best_model/         hardlinked to checkpoint-4 (same inode 32657954)
                    loads via AutoModelForCausalLM.from_pretrained ✓
                    .generate() runs ✓
```

Run details: Qwen2.5-3B-Instruct + MATH-Level345, 8 GPU zero3, 8 steps,
4 evals at steps 2/4/6/8 with rewards 0.053/0.059/0.053/0.059 →
best_metric correctly latched to first peak (step 4).

## The patch

```diff
# trl/trainer/grpo_trainer.py line 2786
-        logs = {**logs, **metrics}
+        # YJ 2026-05-26: in-place update so caller (HF Trainer._maybe_log_save_evaluate)
+        # sees reward/completion metrics in its `metrics` dict — needed for
+        # metric_for_best_model=eval_reward + load_best_model_at_end=true to work.
+        logs.update(metrics)
         super().log(logs, start_time)
```

Behavior delta:
- `super().log(logs, ...)` still receives the same content — wandb /
  console output **unchanged**.
- Caller's `metrics` dict now has the reward keys merged in — so any
  downstream consumer (HF Trainer's `_determine_best_metric`,
  `callback_handler.on_evaluate`, our `BestKeeperCallback`, …) actually
  sees `eval_reward` and friends.

## Why not other approaches

- **Revert to "don't track best at all"**: contradicts the BestKeeperCallback
  design intent and loses the best ckpt on long-running RL.
- **Inline monkey-patch in each `train_*.py`**: violates the
  "trainers self-contained, no shared utility module" convention from
  `projects/co-grpo-dp/CLAUDE.md`, and would need to be copied to 9 files.
- **Bypass via `metric_for_best_model=eval_loss`**: works but uninformative
  for GRPO (eval_loss is ~0 by construction with `beta=0 / loss_type=bnpo`).

The 1-line trl patch is the smallest correct fix and benefits any future
use of GRPO + best-model tracking, not just BestKeeperCallback.

## If you ever pull upstream trl

This is a vendored fork (`trl/` is committed to this repo, not pip-installed),
so upstream pulls would touch this file. If you re-sync from
`huggingface/trl`, just reapply the `.update()` line and the comment marker
(grep for `YJ 2026-05-26`).
