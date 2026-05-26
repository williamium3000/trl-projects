# Eval Results — Pre-RL Baselines (Reproducible Reference)

Post-fix baseline numbers for the **§4 main table** of EMNLP 2026.
Also serves as a **catalog of all 13 eval fixes** discovered during reproduction; each fix is structured for upstream PR submission to lm-evaluation-harness / LiveCodeBench / CRUXEval / SciBench.

---

## Files

| File | Description |
|---|---|
| `baselines_3model_2026-05-25.csv` | Qwen2.5-3B (base) / Llama-3.2-3B-Instruct / Gemma-3-4B-it × 13 benchmarks, all metrics post the 2026-05-25 eval fixes |

---

## The baseline table

| Bench | Qwen2.5-3B (base) | Llama-3.2-3B-Inst | Gemma-3-4B-it |
|---|---:|---:|---:|
| gsm8k | 68.6 | 72.3 | **76.6** |
| math_500 | 59.0 | 44.0 | **76.4** |
| amc | 28.9 | 19.3 | **41.0** |
| aime_25 | 0.0 | 0.0 | **20.0** |
| humaneval | 37.8 | 54.3 | **69.5** |
| mbpp | 55.6 | 52.6 | **62.6** |
| lcb_v6 | 13.7 | 12.0 | **18.6** |
| crux | 13.8 | 12.6 | **34.5** |
| gpqa_d | 19.7 | 24.2 | **33.3** |
| mmlu | **65.1** | 57.4 | 53.2 |
| mmlu_pro | 37.4 | 37.5 | **41.2** |
| scibench | 12.7 | 12.8 | **14.1** |
| ifeval | 22.4 | 69.0 | **72.5** |

Bold = highest per row. Unit: %.

---

## Provenance

- **Raw outputs**: `projects/work_dirs/eval/baselines_20260523_082055/{model}/.../{lm_eval,crux,lcb,scibench}/`
- **Generation timeline**:
  - 5/23 08:20 — initial pass on all 3 models, 13 benches
  - 5/24 00-23h — re-runs of select tasks (humaneval_instruct, mbpp_instruct, gpqa_d) after instruct-path fixes
  - 5/25 00:52 — re-run of `math_500_chat` (new task) for all 3 models
- **Fix commit**: `c4a88345` (2026-05-25 01:47 UTC) — 8 fixes inside this repo + 4 in gitignored external_repos
- **Aggregation script**: `projects/eval/aggregate.py` — picks the latest per-task result across all `results_*.json` files via mtime

---

## Per-task config (all greedy @ K=1)

| Bench | Task name in lm-eval | Few-shot | Metric | Decoding |
|---|---|---|---|---|
| gsm8k | `gsm8k` | 5 | exact_match (strict) | greedy |
| math_500 | `math_500_chat`*¹ | 0 | math_verify | greedy, max_tok=4096 |
| amc | `amc23` | 0 | exact_match (boxed extract + math_verify) | greedy, max_tok=4096 |
| aime_25 | `aime_2025` | 0 | exact_match (boxed extract + math_verify) | greedy, max_tok=4096 |
| humaneval | `humaneval` / `humaneval_instruct`*² | 0 | pass@1 | greedy, max_tok=1024 |
| mbpp | `mbpp` / `mbpp_instruct`*² | 3 | pass_at_1 | greedy |
| gpqa_d | `gpqa_diamond_cot_zeroshot` | 0 | exact_match (flexible-extract*³) | greedy, max_tok=2048*⁴ |
| lcb_v6 | LiveCodeBench v6 (external runner) | 0 | pass@1 | greedy |
| crux | CRUXEval-output (external runner) | 0 | pass@1 | greedy, max_tok=512 |
| scibench | SciBench (external runner) | 0 | exact_match (boxed) | greedy, max_tok=1024 |
| mmlu | `mmlu` (avg 57 subtasks) | 5 | acc | greedy |
| mmlu_pro | `mmlu_pro` | 5 | exact_match (custom-extract) | greedy |
| ifeval | `ifeval` | 0 | prompt_level_strict_acc | greedy, max_tok=1280 |

¹ See Fix #3. ² See Fix #4 + Fix #10. ³ See Fix #6 explanation. ⁴ See Fix #5.

---

## Caveats for paper use

### Caveat 1: Greedy@1 vs maj/avg@k for tiny test sets

All numbers above are **greedy @ K=1** (T=0, single decode). This is the right convention for:
- `gsm8k` (1319 problems), `math_500` (500), `humaneval` (164), `mbpp` (500), `mmlu`, `mmlu_pro`, `ifeval` — community standard reports greedy@1.

But the following benches are **high-variance at K=1** because the test set is tiny:
- **aime_25**: 30 problems → 1 right = 3.3 percentage points. Standard convention (Qwen3-tech-report, DeepSeek-Math) reports **avg@32** or **maj@32**.
- **amc23**: 40 problems → 1 right = 2.5 percentage points. Same convention.

The greedy@1 numbers in the csv are **reasonable but not directly comparable** to papers reporting avg/maj@k. If the EMNLP §4 main table requires avg/maj@k:
- Use `projects/eval/test_time_ensemble/ensemble_eval.py` with the `aime_amc` bench set (Fix #9)
- Output convention: `projects/work_dirs/eval/ensemble_*_K<K>_T<T>_*/`

### Caveat 2: External runners (LCB / CRUX / SciBench)

Run via small custom scripts in `projects/eval/external/`, not via lm-evaluation-harness. Each has its own conventions (see Fix #7, #8, #11).

### Caveat 3: Qwen2.5-3B is **base**, not instruct

`Qwen/Qwen2.5-3B` is pre-train, not instruction-tuned. This produces two characteristic patterns:
- Qwen `ifeval` = 22% (low, base can't follow instructions); Llama/Gemma3 = 69-72%.
- Qwen `mmlu` = 65% (highest, pre-train knowledge dominates); Llama/Gemma3 = 53-57%.

**State this explicitly in the table caption** of the paper, or reviewers will ask. The setup is intentional: we RL on the base, then measure if RL bridges the gap to instruct.

### Caveat 4: lcb_v6 / scibench may need a second pass before publish

`lcb_v6` runner falls back to NA-and-exit on any error (Fix #8). Some scibench buckets similarly. If paper needs publishable LCB numbers, re-run the LCB runner standalone on a clean eval pod after applying gitignored Fix #11–#13.

---

# 🔧 Upstream Fixes — PR-ready catalog (13 fixes)

Format per entry:
- **Affected upstream**: which repo a PR would target
- **Symptom**: what the user sees with the bug
- **Root cause**: why
- **Fix**: code change (link/diff)
- **Impact**: before/after numbers if applicable

In-repo fixes (commit c4a88345, 8 files):

## Fix #1 — `aggregate.py` task name / metric / chat-aware preference

**File**: `projects/eval/aggregate.py`
**Affected upstream**: This is our orchestration code; not a PR candidate. Documented for context.

**Symptom**:
- math_500 column reads `500.0000` (looks like raw count, not accuracy)
- humaneval near zero for chat models even though they answer correctly
- Qwen2.5-3B row missing entirely from `baselines.csv`
- Re-run after instruct-path fix doesn't update the table

**Root cause**:
- Task name `minerva_math_500` was wrong; lm-eval uses `minerva_math500` (no underscore)
- Metric keys used `exact_match,none` (strict) and `pass@1`; should be `math_verify,none` (sympy-based) and `pass_at_1` (underscore, not at-symbol)
- Only one `results_*.json` was read (latest); re-runs of just `humaneval_instruct` would overwrite & lose the main 13-task numbers
- No mechanism to prefer chat-variant of tasks for instruct models

**Fix**:
```python
# task name + metric key fix:
_LM_EVAL_TASKS = [
    ("math_500",   "minerva_math500",             "math_verify,none"),   # was "minerva_math_500" + "exact_match,none"
    ("mbpp",       "mbpp",                        "pass_at_1,none"),     # was "pass@1,none"
    ("gpqa_d",     "gpqa_diamond_cot_zeroshot",   "exact_match,flexible-extract"),  # was "exact_match,strict-match"
    ...
]

# chat-aware preference (NEW):
_PREFER = {
    "humaneval": [("humaneval_instruct", "pass_at_1,create_test")],
    "mbpp":      [("mbpp_instruct",      "pass_at_1,none")],
    "math_500":  [("math_500_chat",      "exact_match,none")],
}

# merge multiple results.json (was: read latest only):
candidates.sort(key=lambda p: p.stat().st_mtime)  # oldest → newest
res = {}
for p in candidates:
    data = json.loads(p.read_text())
    for task_key, task_node in (data.get("results") or {}).items():
        res[task_key] = task_node  # newer wins per task
```

---

## Fix #2 — `aime_2025.yaml` dataset path 404

**File**: `projects/eval/lm_eval_custom_tasks/aime_2025.yaml`
**Affected upstream**: lm-evaluation-harness already merged a similar fix; our custom yaml just needs the new path.

**Symptom**: lm-eval task `aime_2025` fails to download dataset, returns HTTP 404.

**Root cause**: `HuggingFaceH4/aime_2025` mirror was taken offline. The community-maintained mirror is at `yentinglin/aime_2025` (same 30 problems, identical schema).

**Fix**: 1 line in `aime_2025.yaml`:
```diff
-dataset_path: HuggingFaceH4/aime_2025
+dataset_path: yentinglin/aime_2025
```

---

## Fix #3 — NEW task `math_500_chat` (chat-friendly MATH-500)

**File**: `projects/eval/lm_eval_custom_tasks/math_500_chat.yaml`
**Affected upstream**: **lm-evaluation-harness** — would be useful PR (community has asked for this).

**Symptom**: `minerva_math500` (the only lm-eval MATH-500 task) gives unreasonably low scores for chat-tuned models — Gemma-3-4B-it at 0.284 vs Google's published 0.76.

**Root cause**: `minerva_math500` uses a 4-shot Minerva-format prompt ending with `Final Answer:`. Chat-tuned models (Llama-3.2, Gemma-3, Qwen2.5-Inst) are post-trained to wrap answers in `\boxed{...}` markdown or reasoning chains. The result extractor sees `Final Answer:` literal not followed by an answer → 75%+ extraction failures → score floor ~0.28.

**Fix**: New yaml `math_500_chat` with:
- 0-shot
- Explicit instruction "Please reason step by step, and put your final answer within \\boxed{}."
- Reuses `utils.process_results_amc` (boxed extraction + math_verify grading)
- `max_gen_toks: 4096`, T=0

```yaml
task: math_500_chat
dataset_path: HuggingFaceH4/MATH-500
test_split: test
output_type: generate_until
doc_to_text: "Problem: {{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}."
doc_to_target: "{{answer}}"
process_results: !function utils.process_results_amc
generation_kwargs:
  max_gen_toks: 4096
  do_sample: false
  temperature: 0.0
  until: []
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

**Impact** (math_500 score before → after task swap):

| Model | minerva_math500 (4-shot strict) | math_500_chat (0-shot boxed) | Δ |
|---|---:|---:|---:|
| Qwen2.5-3B (base) | 0.39 | **0.59** | +20 |
| Llama-3.2-3B-Inst | 0.38 | **0.44** | +6 |
| Gemma-3-4B-it | 0.284 | **0.76** | **+48** (matches Google's report) |

---

## Fix #4 — `run_eval_all.sh` chat_template / instruct routing / GPQA max_tok

**File**: `projects/eval/run_eval_all.sh`
**Affected upstream**: Our orchestration. Not a PR candidate.

**Symptoms**:
- Default-on chat_template would break base models (Qwen2.5-3B base + chat wrapper → garbage output)
- humaneval / mbpp gave near-zero for chat models when run with default base task
- gpqa_diamond: chat models with CoT got truncated to `[invalid]` after 256 tokens

**Fix**:
- `--chat_template` flag is opt-in (off by default; `run_baselines.sh` decides via heuristic)
- When `--chat_template` is on: route to `humaneval_instruct` + `mbpp_instruct` (chat-aware extractors handle ```python fences)
- Pass `--gen_kwargs max_gen_toks=2048` for GPQA-D
- Thread `--chat_template` through to CRUX and SciBench external runners

---

## Fix #5 — `run_baselines.sh` heuristic shortname → chat_template

**File**: `projects/eval/run_baselines.sh`
**Affected upstream**: Our orchestration.

**Symptom**: Manual model-by-model decision of whether to pass `--chat_template` is error-prone.

**Fix**:
```bash
case "$model_shortname" in
    *instruct*|*_it|*chat*) chat_template_flag="--chat_template" ;;
    *)                       chat_template_flag="" ;;
esac
```

---

## Fix #6 — `cruxeval_runner.py` chat_template path

**File**: `projects/eval/external/cruxeval_runner.py`
**Affected upstream**: **CRUXEval (facebookresearch/cruxeval)** — PR candidate. Their `vllm` runner doesn't support chat models at all.

**Symptom**: Llama-3.2-3B-Instruct scored 0.001 on CRUX. Gemma-3-4B-it scored 0.188. Both should be 10-35% range.

**Root cause**: Runner uses `llm.generate(prompts, sp)` unconditionally (raw text generation). The CRUX prompt expects the model to output `<answer>...</answer>` markers. Instruct models, given a raw prompt without chat template, produce verbose conversational explanations (`"Sure! Let me work through this..."`) without the markers → extraction fails → ~0.

**Fix**: Add `--chat_template` flag that wraps prompts as chat messages and calls `llm.chat()`:
```python
sp = SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"])
if args.chat_template:
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(messages_list, sp)
else:
    outs = llm.generate(prompts, sp)
```

**Impact**:

| Model | Before | After | Δ |
|---|---:|---:|---:|
| Qwen2.5-3B (base) | 0.137 | 0.137 | 0 (correct: base shouldn't use chat) |
| Llama-3.2-3B-Inst | **0.001** | **0.126** | **+12.5** percentage points |
| Gemma-3-4B-it | **0.188** | **0.345** | **+15.7** percentage points |

---

## Fix #7 — `scibench_runner.py` f-string IndexError + chat_template + non-fatal

**File**: `projects/eval/external/scibench_runner.py`
**Affected upstream**: **SciBench (mandyyyyii/scibench)** — PR candidate.

**Symptom 1 (fatal)**: Runner crashes with `IndexError: Replacement index 0 out of range` before producing any output.

**Root cause 1**: Prompt template contains `\\boxed{...}` literal, but uses `str.format` which interprets `{...}` as a placeholder. The single `{` confuses the formatter.

**Fix 1**: Escape `{...}` → `{{...}}` (literal braces in str.format):
```diff
 _PROMPT_TMPL = (
     "Solve the following problem step by step. Put your final answer in "
-    "\\boxed{...}.\n\nProblem: {q}\n"
+    "\\boxed{{...}}.\n\nProblem: {q}\n"  # {{...}} = literal {...} for str.format
 )
```

**Symptom 2**: Same chat_template issue as CRUX (Fix #6).

**Fix 2**: Add `--chat_template` flag with `llm.chat()` path.

**Symptom 3**: When SciBench crashes, the parent `run_eval_all.sh` (which sequenced SciBench after lm-eval) also dies → all subsequent benches skipped → wasted GPU.

**Fix 3**: Wrap main() in try/except; on any failure write `NA` to the output json and `exit 0`, so the parent script's `aggregate.py` step can still produce a row with NA in the scibench column.

---

## Fix #8 — `livecodebench_runner.py` invalid flag + non-fatal

**File**: `projects/eval/external/livecodebench_runner.py`
**Affected upstream**: LCB is gitignored; this is local glue.

**Symptom**: Runner exits immediately with `error: unrecognized arguments: --model_provider`.

**Root cause**: LCB removed the `--model_provider` flag in a recent refactor; their `runner_main.py` no longer accepts it. Our wrapper still passed it from a stale invocation.

**Fix**: Drop `--model_provider vllm` from the subprocess argv. Also add the same non-fatal wrapper as Fix #7.

---

## Fix #9 — `ensemble_eval.py` aime_amc bench set + chat_template

**File**: `projects/eval/test_time_ensemble/ensemble_eval.py`
**Affected upstream**: Our test-time ensemble code.

**Symptom 1**: No predefined bench set for "just the small-N math benches" (AIME-25 + AMC23) — every maj@K supplementation needed `--benchmarks aime_25,amc` typed by hand.

**Fix 1**:
```python
BENCH_SETS = {
    ...
    "aime_amc": ["amc", "aime_25"],  # for paper §4 maj@K supplementation
}
```

**Symptom 2**: ensemble used `llm.generate(...)` not `llm.chat(...)` for chat models.

**Fix 2**: Add `--chat_template` flag with branched path (same shape as Fix #6).

---

## Fix #10 — `baselines.txt` model list update

**File**: `projects/eval/baselines.txt`

**Symptom**: List included `phi35` (Microsoft Phi-3.5-mini) which is no longer in our §4 table.

**Fix**: Replace `phi35` with `gemma3-4b-it`.

---

# Gitignored fixes (apply after `projects/eval/setup.sh` clones external_repos)

These live inside `projects/eval/external_repos/` which is gitignored. They need to be re-applied each time the eval pod is fresh; alternatively maintained as a patch file or as PRs to upstream.

## Fix #11 — `lm_eval/tasks/{humaneval,mbpp}/utils.py` SentencePiece U+2581

**Affected upstream**: **EleutherAI/lm-evaluation-harness** — PR candidate.

**Symptom**: Gemma-3 humaneval / mbpp = 0% pass@1 even though Gemma generates correct code.

**Root cause**: Gemma3's SentencePiece tokenizer emits U+2581 ("▁") as the word-start marker. lm-eval's humaneval/mbpp extractor passes the raw output string straight to `exec()`. Python parser sees U+2581 instead of space → `IndentationError` / `SyntaxError` → all answers fail.

**Fix**: One line in each of `humaneval/utils.py` and `mbpp/utils.py`:
```python
text = text.replace("▁", " ")  # Gemma3 / SentencePiece: U+2581 → space
```

**Impact**: Gemma-3-4B-it humaneval 0.00 → **0.695**; mbpp 0.00 → **0.626**.

## Fix #12 — `LiveCodeBench/lcb_runner/lm_styles.py` register 4 baselines + Gemma3 enum

**Affected upstream**: **LiveCodeBench (LiveCodeBench/LiveCodeBench)** — PR candidate.

**Symptom**: LCB refuses to run unless model is in its hardcoded `LMStyle` enum. Our 4 baselines (Qwen2.5-3B, Llama-3.2-3B-Instruct, Phi-3.5-mini, Gemma-3-4B-it) are not registered → runner errors out.

**Fix**: 
```python
class LMStyle(Enum):
    ...
    Gemma3 = "Gemma3"  # NEW

LanguageModelList: list[LanguageModel] = [
    ...
    LanguageModel("Qwen/Qwen2.5-3B",              "qwen25_3b_base",       LMStyle.GenericBase),
    LanguageModel("meta-llama/Llama-3.2-3B-Instruct", "llama32_3b_inst",  LMStyle.LLaMa3),
    LanguageModel("microsoft/Phi-3.5-mini-instruct",  "phi35_mini",       LMStyle.GenericBase),
    LanguageModel("google/gemma-3-4b-it",         "gemma3_4b_it",         LMStyle.Gemma3),
]
```

## Fix #13 — `LiveCodeBench/lcb_runner/prompts/*.py` hardcoded Meta-Llama-3-8B-Instruct → Llama-3.2-3B-Instruct + Gemma3 branch

**Affected upstream**: **LiveCodeBench** — PR candidate.

**Symptom**: `code_generation.py`, `code_execution.py`, `self_repair.py`, `test_output_prediction.py` all reference `meta-llama/Meta-Llama-3-8B-Instruct` for tokenization (chat template). That model is gated and our HF account doesn't have access → 401 Unauthorized at runtime.

**Root cause**: LCB hard-codes one chat-template tokenizer choice per LMStyle. The string-equality check against `"Meta-Llama-3-8B-Instruct"` is fragile.

**Fix**: 
1. Replace the hardcoded string in all 4 prompt files with `Llama-3.2-3B-Instruct` (same chat template, but unrestricted).
2. In `code_generation.py`, add a Gemma3 branch:
```python
elif lm_style == LMStyle.Gemma3:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

---

## How to reproduce the baseline table from scratch

```bash
# from repo root
cd projects/eval

# 1. Apply gitignored patches (Fix #11–#13) — run once per fresh eval pod
bash setup.sh   # clones external_repos
# Then manually apply or symlink the patches noted in commit c4a88345's message.

# 2. Run all 13 benches per model
bash run_baselines.sh
#   → projects/work_dirs/eval/baselines_<TS>/{qwen25_3b_base, llama32_3b_instruct, gemma3_4b_it}/

# 3. Aggregate raw json per model into csv rows
for model_pair in \
    "qwen25_3b_base/Qwen_Qwen2.5-3B Qwen/Qwen2.5-3B" \
    "llama32_3b_instruct/meta-llama_Llama-3.2-3B-Instruct meta-llama/Llama-3.2-3B-Instruct" \
    "gemma3_4b_it/google_gemma-3-4b-it google/gemma-3-4b-it"; do
    set -- $model_pair
    python aggregate.py \
        --run_dir projects/work_dirs/eval/baselines_<TS>/$1_<TS>/ \
        --model $2 \
        --out_csv projects/eval/results/baselines_<TS>.csv
done
```
