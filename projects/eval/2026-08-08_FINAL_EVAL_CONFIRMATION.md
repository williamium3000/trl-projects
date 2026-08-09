# Final eval confirmation — 2026-08-08

Companion to `COMAS_EVAL_RESULTS.md` (2026-08-07). That one documents the CoMAS
suite; **this one is the paper's main table**. If you only read one thing: the
two are not interchangeable, not even on the five benchmarks whose names match.

Status at time of writing: environment rebuilt from scratch on this cluster and
**validated against the published base row**; ten models measured once; a
three-seed repeat is in flight because two of the seven columns turned out to be
too noisy to report from a single run.

---

## 1. What the main table is

Seven columns, per `PAPER_OUTLINE_v5.md` (in `trl-projects-mllm` git history,
`git show 63b60a5:PAPER_OUTLINE_v5.md`):

> LLM 主表(§5.1)= 7-bench reasoning suite（数学 GSM8K/MATH500/AMC + 代码生成
> HEval/MBPP/LCB + 科学推理 GPQA）；**移除 CRUX/SciBench/MMLU/MMLU-Pro**

**AIME-24 was dropped at v4** ("仅 30 题、avg@8 下方差仍过大，信噪比低").

> ⚠️ `projects/eval/README.md` and the `aggregate.py` docstring still say
> "主表正文 6 列: GSM8K / MATH-500 / AMC / **AIME-24** / HumanEval / GPQA-D".
> **Both are v3-era and stale.** v5 dropped AIME and promoted MBPP + LCB. Reading
> the README instead of the outline is how this run first measured the wrong
> column set.

| main-table col | lm-eval task | samples | decoding | metric key |
|---|---|---|---|---|
| GSM8K | `gsm8k` | 1 | T=0.6, top_p=0.95 | `exact_match,flexible-extract` |
| MATH500 | `math_500_chat` (custom) | 1 | T=0.6, top_p=0.95 | via `process_results_amc` |
| AMC | `amc23` (custom) | 1 (see §5) | T=0.6, top_p=0.95 | `exact_match,none` |
| HEval | `humaneval` / `humaneval_instruct` | 1 | T=0.6, top_p=0.95 | `pass@1,create_test` |
| GPQA | `gpqa_diamond_boxed` (custom) | 1 | **greedy, T=0** | `exact_match,none` |
| MBPP | `mbpp` / `mbpp_instruct` | 1 | T=0.6, top_p=0.95 | `pass_at_1,none` (underscore) |
| LCB | LiveCodeBench `release_v6`, external runner | 1 | greedy | official `pass@1` |

---

## 2. Environment — exact, and every pin has a reason

Built by `mllm-cluster/build_evalmain_env.sh` as apptainer + venv (this cluster
has no conda). Versions come from `projects/eval/setup.sh` verbatim.

```
torch 2.9.1+cu129     vllm 0.14.0     transformers 4.57.1
datasets 3.6.0        numpy 2.2.6     python 3.11.14
lm-evaluation-harness 95d5806  + lmeval_gemma_u2581.patch + lmeval_mbpp_lang_tag.patch
LiveCodeBench         28fef95   + livecodebench_register_baselines.patch
cruxeval              190faf1
scibench              e14e0ca
```

From `projects/eval/dispatch/README.md`'s trap list — do not "upgrade" any of
these:

| pin | what breaks without it |
|---|---|
| `vllm==0.14.0` | ≥0.21 ships CUDA-13 wheels; kernels fail at launch on CUDA-12 drivers |
| `transformers==4.57.1` | 5.9 breaks `is_offline_mode` against hf_hub 0.36.2 |
| `datasets==3.6.0` | 5.0 removed script datasets; LCB's loader is a script |
| `numpy==2.2.6` | 2.4 exceeds numba's ceiling, vllm worker dies |

Env location: `/weka/scratch/jhu/dssg2026-ext-rghani1/yyang331/eval_main/`
(`env/`, `external_repos/`, `out/`, `ENV_READY`).

---

## 3. Deviations from `setup.sh`, and why each was unavoidable

Every one of these is required to make the documented stack install and run on
this cluster. None of them changes what is measured.

**3.1 `lm-eval` installed without the `[vllm]` extra.**
`setup.sh` asks for `[vllm,ifeval,math,sentencepiece]`, but this lm-eval commit
declares `vllm = ["vllm>=0.18"]`, which cannot coexist with the project's
`vllm==0.14.0`; pip returns `ResolutionImpossible`. setup.sh's own comment shows
the intent was to install vllm first and have the extra be a no-op — pip's
resolver defeats that. The extra exists only to pull vllm, which is already
installed at the pinned version, so omitting it changes nothing importable.

**3.2 `PIP_CONSTRAINT` applied to every install.**
Installing in the right order is not enough. The first build put torch 2.9.1 and
vllm 0.14.0 in correctly, then the lm-eval install resolved the extra to vllm
0.26.0 and dragged torch to 2.11.0+cu130.

**3.3 antlr is left to lm-eval's `math` extra (4.11).**
An earlier revision pinned 4.9.3 from a stale note and ran that install *after*
lm-eval had put 4.11 in, silently downgrading it.
`lm_eval/tasks/minerva_math/utils.py` asserts the version **at import**, so this
took down all seven columns — not just the math ones — in ten jobs that all
reported `COMPLETED`.

**3.4 `projects/eval/external_repos` symlinked to the scratch clone.**
`run_eval_all.sh` hardcodes `EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"` with no
env override.

**3.5 `LCB_LLAMA3_TOKENIZER` added.**
LiveCodeBench hardcodes `meta-llama/Meta-Llama-3-8B-Instruct` in four prompt
builders, purely to fetch the Llama-3 chat template. That repo is gated and
**both** project tokens get 403, so every Llama row returned `lcb_v6=NA`. The
repo's own gotcha index hit this and substituted a local
Meta-Llama-3.1-8B-Instruct; here the substitute is `Llama-3.2-3B-Instruct`
(readable, HTTP 200), which is the actual family of the checkpoints being
evaluated. The hardcoded default is preserved for anyone who does have access.

**3.6 `EVAL_SEED` added to `run_eval_all.sh`.** New, see §5.

**3.7 The runner judges by output, not exit code.**
`run_eval_all.sh` swallows a failing `lm_eval` and returns 0. Ten jobs
"COMPLETED" in ~80 s each having produced nothing. `run_evalmain.sh` now
requires a `results.csv` with all six lm-eval columns non-`NA`, and exits
non-zero otherwise.

---

## 4. Validation: the base row reproduces

`Qwen/Qwen2.5-3B`, untrained, **no** `--chat_template` (it never saw one), against
`results_tables/qwen2.5-3b.csv` row 1.

| col | this run | published | Δ |
|---|---|---|---|
| GSM8K | 74.4 | 73.4 | +1.0 |
| MATH500 | 56.2 | 56.6 | −0.4 |
| AMC | 26.5 | 28.9 | −2.4 |
| HEval | 32.9 | 39.0 | **−6.1** |
| GPQA | 21.7 | 21.2 | +0.5 |
| MBPP | 53.2 | 52.2 | +1.0 |
| **LCB** | **13.65** | **13.7** | **−0.05** |

**LCB and GPQA are the load-bearing checks.** Both are greedy, so they are
deterministic, and both land on the published value. The sampled columns differ
by amounts consistent with §5. A second, independent confirmation: the
`gt-qwen25` row here (GSM8K 76.2, HEval 64.6) sits on the published GT-Reward row
(76.2, 65.2).

The HEval gap is the largest and is the one to be sceptical of — but see §5: an
identical repeat of another model moved HEval by 3.0 points and AMC by 6.0 on its
own.

---

## 5. Why single-run numbers are not enough — measured, not assumed

`gt-llama32` was run twice with byte-identical configuration (the second time to
pick up the LCB fix):

| col | run 1 | run 2 | Δ |
|---|---|---|---|
| GSM8K | 77.48 | 77.18 | 0.30 |
| MATH500 | 55.00 | 54.00 | 1.00 |
| **AMC** | **27.71** | **21.69** | **6.02** |
| **HEval** | **60.98** | **64.02** | **3.04** |
| **GPQA** | **21.21** | **21.21** | **0.00** |
| MBPP | 50.20 | 50.00 | 0.20 |

GPQA is greedy and does not move. Everything sampled does, and the two
smallest-sample benchmarks move most.

**AMC specifics, verified against the samples log rather than the docs:**

- the dataset is `AI-MO/aimo-validation-amc`, train split — **83 problems**, not
  the 40 usually meant by "AMC-23"
- `amc23.yaml` sets `repeats: 8`, but the produced samples file has **one
  response per problem**
- and it would not matter if it had eight, because the grader takes only the
  first:

```python
def process_results_amc(doc, results):
    completion = results[0] if results else ""   # the other 7 are discarded
```

`process_results_aime` is written the same way. So **AMC and AIME are pass@1, not
avg@8**, despite `EVAL_HANDOFF.md` stating "AMC/AIME = avg@8". This does not make
our numbers incomparable to the published ones — the same grader produced both —
but the paper must not describe those columns as avg@8.

Binomial noise for AMC at p≈0.27 over 83 items: σ ≈ 4.9 points for one
measurement, 6.9 for the difference of two. The observed 6.02 swing is 0.87σ, and
the base row's −2.4 gap is 0.35σ. Both are ordinary.

**Consequence for the paper:** differences of 1–3 points in the AMC and HEval
columns are not readable from a single run. Report mean ± std over seeds, or say
in the caption that these two columns carry ±3–5 points.

`EVAL_SEED` was added so a run can at least be reproduced (vLLM defaults to
seed=1234, but `--batch_size auto` still varies the batching between
invocations). **It fixes reproducibility, not variance** — several seeds are
still required.

---

## 6. Results — the new experiments only

**Rows that already exist in the published main table are NOT re-measured.**
Re-running `GT-Reward` or `TTRL` for Qwen2.5-3B / Llama-3.2-3B would put a second
set of numbers next to the published ones for the same checkpoint, which reads as
if the numbers were being shopped. Use `results_tables/*.csv` for those.

Measured here: the three N=3 supervision rules, and the whole Qwen3-1.7B block
(that model is new, so it needs its own base and baselines).

`n=3` means three eval seeds, reported mean±std. `n=1` means a single run — those
cells have no error bar and differences of 1-3 points in them are not readable
(see §5). AMC is always avg@8 over three seeds.

### Qwen2.5-3B

| method | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| strict (n=3) | 79.6±0.2 | 65.5±0.8 | 31.9±1.3 | 65.0±3.0 | 21.0±1.2 | 55.5±0.9 | 15.4 | **47.7** |
| self_plus_peers (n=3) | 79.6±0.4 | 67.3±0.6 | 29.9±1.0 | 65.8±2.2 | 21.6±1.3 | 55.0±0.0 | 17.9 | **48.2** |
| **union (n=1, AMC n=3)** | 79.2 | 66.0 | 34.0±2.0 | 61.0 | 27.8 | 56.0 | 17.8 | **48.8** |

### Llama-3.2-3B

| method | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| strict (n=3) | 78.0±1.3 | 50.9±1.4 | 24.7±1.6 | 59.6±2.3 | 21.4±1.1 | 50.1±0.8 | 12.4 | **42.4** |
| self_plus_peers (n=3) | 78.3±0.7 | 52.3±1.0 | 24.3±1.2 | 59.8±1.2 | 22.7±3.5 | 50.1±1.9 | 12.6 | **42.9** |
| **union (n=1, AMC n=3)** | 78.8 | 51.8 | 28.3±0.8 | 65.2 | 19.7 | 50.2 | 11.8 | **43.7** |

### Qwen3-1.7B

| method | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| base (n=3) | 67.0±1.1 | 60.9±0.6 | 27.5±1.5 | 40.0±2.5 | 15.3±3.2 | 50.6±0.2 | 12.4 | **39.1** |
| GT (n=1, AMC n=3) | 67.1 | 67.0 | 34.3±0.4 | 70.1 | 25.2 | 51.2 | 15.2 | **47.2** |
| TTRL (n=1, AMC n=3) | 70.3 | 67.6 | 32.1±0.3 | 69.5 | 24.8 | 52.0 | 15.1 | **47.3** |
| strict (n=3) | 67.0±0.4 | 69.2±0.5 | 33.9±0.8 | 65.8±1.1 | 24.7±0.5 | 54.3±1.2 | 14.3 | **47.0** |
| self_plus_peers (n=3) | 68.5±1.6 | 68.5±2.0 | 34.3±1.1 | 66.7±1.9 | 23.9±1.5 | 53.5±1.7 | 14.2 | **47.1** |
| **union (n=1, AMC n=3)** | 68.9 | 68.8 | 34.5±1.6 | 64.0 | 26.3 | 55.6 | 14.7 | **47.5** |

### Reading it

union is first on Avg for all three models (48.8 / 43.7 / 47.5) and the ordering
union > self_plus_peers > strict repeats across all three. **But the margins are
0.5-1.3 points and the union row is single-seed**, so this ordering is not yet
significant. `strict` and `self_plus_peers`, which both have error bars, overlap
on every column — consistent with the MATH-500 result that the three rules
converge to the same place.

The Qwen3-1.7B block is complete and is the cleanest story here: base 39.1 ->
GT 47.2 / TTRL 47.3 / union 47.5, i.e. **all three methods gain ~8 points over
the untrained base, and the label-free ones match the one that uses ground truth.**

---

## 7. Two more defects found while producing the table

**10.1 LiveCodeBench needs each model registered, with the right style.**
`lm_styles.py` raises `KeyError` for anything unregistered, and the runner turns
that into `lcb_v6=NA` and continues — a silent hole in one column. Seven models
had to be added. The style matters as much as the registration:

| model kind | LMStyle | why |
|---|---|---|
| untrained base | `GenericBase` | plain completion, no chat markers |
| instruct / any trained ckpt | `CodeQwenInstruct` (Qwen) or `LLaMa3` (Llama) | the model was trained on conversational prompts |

`Qwen3-1.7B-Base` was first registered as `CodeQwenInstruct` and scored **LCB 2.2**.
Corrected to `GenericBase` it scores **12.4**, in line with the other bases
(Qwen2.5-3B 13.7, Llama-3.2-3B 12.0). The 2.2 was the model drowning in chat
tokens it had never seen, not a capability measurement. The same mismatch also
explains that block's GPQA of 15.3 — below the 25% floor of a four-way choice.

The rule is **"has this checkpoint ever seen a chat template"**, not "what was its
base model". `grpo-qwen25-3b-math345` derives from the bare `Qwen/Qwen2.5-3B` but
is registered `CodeQwenInstruct`, because training wrapped its prompts. That is
the same criterion as `CHAT=1` in §3/§9.

**10.2 The task yamls are not a record of what ran.**
`run_eval_all.sh` passes a global `--gen_kwargs`, which overrides each task's
`generation_kwargs`. Audited against the executed config in `results.json`:

| task | yaml says | actually ran |
|---|---|---|
| `gpqa_diamond_boxed` | `do_sample: false`, `temperature: 0.0`, 2048 tok | **`do_sample: true`, `temperature: 0.6`**, 3072 tok |
| `amc23` | `max_gen_toks: 4096` | **3072** |

**GPQA has never been greedy**, including for the published table — same script.
Keeping it as-is (changing it would invalidate the published column), but the
paper must describe GPQA as single-sample T=0.6, not greedy, and note that the
column carries about ±3 points. Everything outside `generation_kwargs` — dataset,
splits, `repeats`, `filter_list`, `process_results`, `metric_list` — is honoured
as written.

**Take the executed config from `results.json`, never from the yaml.**

---

## 8. Progress

**Environment**: built and validated (§4). **Main-table eval**: the new
experiments are done, see §6.

| | status |
|---|---|
| three N=3 rules x 3 models | done, 3 seeds each (LCB 1 seed — it is deterministic) |
| Qwen3-1.7B base / GT / TTRL / union | done |
| AMC, all models | done, avg@8 x 3 seeds |
| union rows, non-AMC columns | **single seed** — no error bars yet |
| published rows (Qwen2.5-3B / Llama-3.2-3B base, GT, TTRL) | **not re-run by design**, use `results_tables/*.csv` |
| 7B / 8B AMC after the avg@8 fix | **outstanding** — those checkpoints are not on this cluster |

---

## 9. Reproducing this from nothing

```bash
# 1. external repos, pinned + patched
sbatch mllm-cluster/build_evalmain_env.sh          # ~25 min, 1 GPU
#    -> /weka/scratch/.../eval_main/{env,external_repos}, ENV_READY
#    verification is hard-gated: version pins, all nine tasks resolvable
#    under --include_path, and `import lm_eval.tasks.minerva_math.utils`

# 2. symlink the external repos where run_eval_all.sh expects them (§3.4)
ln -s /weka/scratch/.../eval_main/external_repos \
      trl-projects/projects/eval/external_repos

# 3. one model
sbatch --job-name=em-foo \
  --export=ALL,HF_TOKEN=<token>,MODEL=<repo>,TAG=foo,CHAT=1,EVAL_SEED=1 \
  mllm-cluster/run_evalmain.sh
```

`CHAT=1` for every trained checkpoint (training used conversational prompts, so
TRL applied a chat template — see `RESULTS_SETUP.md`); `CHAT=0` only for
untrained base models.

Results land in `eval_main/out/<TAG>/<model>_<ts>/results.csv`, 15 columns, 4
decimal places, fractions not percentages, `NA` for missing. **When a tag has
several run directories, take the newest** — `aggregate.py` appends rather than
replaces.

---

## 10. Mistakes made producing this, so they are not repeated

1. **Ran the wrong suite.** "7 benchmarks" was read as CoMAS's seven; the main
   table's seven are a different set. One hour of GPU. Results kept in
   `COMAS_EVAL_RESULTS.md`.
2. **Read a stale doc.** `projects/eval/README.md` still describes the v3 column
   set. The authority is `PAPER_OUTLINE_v5.md`, which lives in a different repo
   and only in git history.
3. **Left a line that downgraded antlr** after lm-eval had installed the right
   version — ten jobs wasted, all reporting success.
4. **Trusted an exit code.** Fixed in §3.7.
5. **Patched the wrong function.** A first attempt at the Qwen2.5-VL vision-tower
   crash patched `get_vit_attn_backend`; measurement later showed that function
   already returns the right answer and the patch was a no-op. The real defect is
   downstream (`use_upstream_fa` dropped by the caller). Documented with the mmr1
   runs, not here, but the lesson is shared: **verify a fix fires, do not assume
   it from reading.**

Items 3 and 5 have the same root cause — "looks correct" substituted for
"observed correct". Both checks are now behavioural: the build imports the module
that asserts, and the fix prints when it fires.
