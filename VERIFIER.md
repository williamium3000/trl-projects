# Verifier Convention (parser + grader) for trl-projects

This repo's projects (`mllm-co-grpo-dp`, `co-opd`, `co-grpo-dp`, `un-grpo-maj`)
all train and evaluate against math-style or VLM-counting answers. Every such
task needs two pieces:

1. **Parser** — pull the answer out of a model completion (`\boxed{}` for math,
   `<answer>...</answer>` for R1-V VLM).
2. **Grader** — decide whether `pred ≡ gold` mathematically (`1/2 ≡ 0.5`).

**Decision (verified 2026-05-17): use one shared parser+grader across all
projects, sourced from `qwen-sympy` (`verifiers/qwen/`), with one local fix.**
This lets every project run inside the single `marti-parity` conda env and
removes the need for the bucket-A / bucket-B env split.

---

## TL;DR

| Concern | Choice | Why |
|---|---|---|
| Grader (math equivalence) | `verifiers/qwen/math_grade.py:grade_answer` (PRM800K / Hendrycks lineage, shipped under the `qwen/` folder) | 100% pass on 320+ real golds across 9 datasets; byte-aligned across `co-grpo-dp` / `co-opd` / `mllm-*` |
| Parser for math `\boxed{}` | **Balanced-bracket regex (10 lines)**, NOT `qwen.extract_answer` | `qwen.extract_answer` eats single-letter latin variables (`a`, `c`, `g`) inside latex — observed on Minerva (12.5% false-negative) and AMO-Bench (20%) |
| Parser for R1-V VLM | `verifiers/math_verify_wrapper.py:extract_answer_tag` (`<answer>...</answer>` regex, last match) | CLEVR / GEOQA solutions are tag-wrapped; works as-is |
| HF `math_verify` package | **Do NOT install** | Hard antlr version conflict with our pinned `latex2sympy2` (see §3) |
| Env | Single `marti-parity` env (clone of `marti` conda env, no math-verify) | All datasets pass with qwen-sympy grader |

---

## 1. Empirical evidence (single env covers everything)

Test harness: stream-loaded 30–40 golds from each of 9 datasets, built
`"...$\boxed{<gold>}$."` completions plus four real-world aliases (`$$...$$`
wrap, trailing period, `\dfrac`→`\frac`, multi-boxed). Parser = balanced
bracket extractor. Grader = `qwen.grade_answer`.

| Dataset | Used by | Format | Perfect-match | All aliases combined |
|---|---|---|---|---|
| CLEVR-Counting (`leonardPKU/clevr_cogen_a_train`) | mllm-co-grpo-dp | `<answer> N </answer>` | 40/40 = 100% | 120/120 = 100% |
| GEOQA-R1V-Train-8K (`leonardPKU/GEOQA_R1V_Train_8K`) | mllm-co-grpo-dp | `<answer> N° </answer>` | 40/40 = 100% | 119/120 = 99.2%¹ |
| MATH-500 (`HuggingFaceH4/MATH-500`) | co-opd eval, co-grpo-dp eval | `\boxed{X}` | 40/40 = 100% | 124/124 = 100% |
| AIME24 (`HuggingFaceH4/aime_2024`) | co-opd eval | `\boxed{N}` | 30/30 = 100% | 90/90 = 100% |
| AIME25 (`yentinglin/aime_2025`) | co-opd eval | `\boxed{N}` | 30/30 = 100% | 90/90 = 100% |
| AMC23 (`math-ai/amc23`) | co-opd eval | `\boxed{N}` | 40/40 = 100% | 120/120 = 100% |
| Minerva (`math-ai/minervamath`) | co-opd eval | `\boxed{<physics latex>}` | 40/40 = 100% | 124/124 = 100% |
| AMO-Bench (`meituan-longcat/AMO-Bench`) | co-opd eval | mixed (boxed + descriptive) | 40/40 = 100% | 130/130 = 100% |
| HMMT25 (`MathArena/hmmt_feb_2025`) | co-opd eval | `\boxed{X}` | 30/30 = 100% | 100/100 = 100% |

¹ The single GEOQA fail is `40度` (Chinese character `度` for "degree", not
the unicode `°`). Fix: add `("度", "")` to the `_UNICODE_FIXUPS` list in
`verifiers/math_verify_wrapper.py`. Verifier choice is unrelated.

**Conclusion: one parser + one grader covers every training and eval dataset
the repo uses.**

---

## 2. The parser change (do this when you wire up a new math project)

Replace any call to `qwen.qwen_math_parser.extract_answer` in a `\boxed{}`
extraction path with this:

```python
def extract_boxed_balanced(s: str) -> str | None:
    """Return the content of the LAST \\boxed{...} in `s`, balancing braces.

    Handles nested braces (e.g. \\boxed{\\frac{a}{b}}) which a naive regex
    `\\boxed\\{[^{}]*\\}` cannot. Returns None if no \\boxed{ is present or
    the braces never close.
    """
    pos = s.rfind("\\boxed{")
    if pos < 0:
        return None
    start = pos + len("\\boxed{")
    depth, i = 1, start
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return s[start:i-1] if depth == 0 else None
```

### Why `qwen.extract_answer` is unsafe for our data

`verifiers/qwen/qwen_math_parser.py:extract_answer` applies a stack of
"cleanup" heuristics (de-spacing, `\left`/`\right` stripping, unit-word
removal). On latex-heavy physics and olympiad strings these heuristics chew
through single-letter latin variables and parenthesis content. Real observed
failures on the Minerva dataset:

| Gold (truncated) | What `qwen.extract_answer` returned |
|---|---|
| `\frac{2 \pi c^{2} R^{2}}{\lambda^{5}\left[e^{h c /(\lambda k T)}-1\right]^{2}}` | `\frac{2\pi^{2}R^{2}}{\lambda^{5}[e^{\lambdakT)}-1]^{2}}` (kills `c`, `h`) |
| `\frac{a M^{1 / 3}}{G M^{2 / 3}+b}` | `\frac{^{1/3}}{GM^{2/3}+b}` (kills `a` and one `M`) |
| `\left[P_{0}^{2/5}-\frac{2}{5} g K^{-3/5} z\right]^{5/2}` | `[P_{0}^{2/5}-\frac{2}{5}^{-3/5}z]^{5/2}` (kills `g`, `K`) |
| `-1./3` | `\frac{-1}{3}` (eagerly rewrites format) |

`identity_grade(gold, gold) = 100%` in the same run, proving the grader is
fine and the parser is the bug.

### Why the balanced regex is enough

In every dataset we use, the canonical wrapper is one of:

- `<answer>X</answer>` (R1-V) — already handled by `extract_answer_tag`.
- `\boxed{X}` (math) — `extract_boxed_balanced` returns `X` verbatim.

The grader then handles all the equivalence work (`\dfrac`→`\frac`,
`1/2`→`0.5`, latex parsing, sympy simplify). The parser does **not** need to
reformat or clean — it just needs to return the boxed substring intact.

---

## 3. Why HF `math_verify` cannot be installed alongside `qwen-sympy`

**Hard antlr version conflict, no workaround.**

### 3.1 The lock chain

- `qwen-sympy` depends on `latex2sympy2==1.9.1`, which is built against
  ANTLR4 runtime **4.7.2** (it is pinned by the marti env's lockfile).
- HF `math_verify` (any version we have access to, current 0.9.0) depends on
  `latex2sympy2_extended==1.11.0`.
- `latex2sympy2_extended==1.11.0` wheel metadata pins
  `antlr4-python3-runtime>=4.9.3,<=4.13.2`.

Both packages register the same top-level module `antlr4`. **pip allows
exactly one version of `antlr4-python3-runtime` per environment.**

### 3.2 Evidence: pip dry-run forces an antlr upgrade

```
$ pip install --dry-run math-verify     # in marti env (has antlr 4.7.2)
Collecting math-verify
Collecting latex2sympy2_extended==1.11.0 (from math-verify)
Collecting antlr4-python3-runtime<=4.13.2,>=4.9.3 (from latex2sympy2_extended==1.11.0->math-verify)
Would install antlr4-python3-runtime-4.13.2 latex2sympy2_extended-1.11.0 math-verify-0.9.0
```

→ Installing `math-verify` **upgrades** `antlr4-python3-runtime` from 4.7.2
to 4.13.2. After the upgrade, `latex2sympy2 1.9.1`'s parser is loaded against
the wrong ANTLR runtime; in practice this manifests as cryptic errors inside
qwen's grade path (parser ATN format mismatch).

### 3.3 Evidence: low-antlr + math_verify is explicitly blacklisted

To rule out "downgrade antlr after the fact," I built an isolated env with
`antlr4-python3-runtime==4.7.2` and force-installed
`latex2sympy2_extended==1.11.0` and `math-verify==0.9.0` with `--no-deps`,
then ran `from math_verify import parse, verify`:

```
ImportError: Unsupported ANTLR version 4.7.2, only 4.9.3, 4.11.0, and 4.13.2
             runtime versions are supported.
```

This is raised by `latex2sympy2_extended/antlr_parser.py:18` — the package
ships an explicit allowlist. ANTLR 4.7.2 is **not on it** and will never be:
the generated lexer/parser code uses ATN serialization v3 (4.9+ format).

### 3.4 Summary of the deadlock

|  | needs antlr | can it run? |
|---|---|---|
| `qwen-sympy` (`latex2sympy2 1.9.1`) | 4.7.2 only | ✅ in marti env |
| HF `math_verify` (`latex2sympy2_extended 1.11.0`) | 4.9.3 / 4.11.0 / 4.13.2 only | ❌ in marti env |
| Both at once in one env | impossible | ❌ |

**Therefore**: a single env that runs everything must use `qwen-sympy` and
must NOT install `math-verify`.

---

## 4. Operating rules

### 4.1 Do

- Use `verifiers/qwen/math_grade.py:grade_answer` as the single grader across
  projects.
- Use `extract_boxed_balanced` (§2) for any `\boxed{}` extraction.
- Use `verifiers/math_verify_wrapper.py:extract_answer_tag` for R1-V
  `<answer>` extraction.
- When adding a new dataset, add a sampled gold row to the stress harness
  (sample script under `/tmp/stress_full.py` during the 2026-05-17 audit;
  copy into `tests/` if you want it permanent) and confirm 100% identity
  pass before training against it.

### 4.2 Do NOT

- **Do NOT `pip install math-verify` or `latex2sympy2_extended`** into the
  marti env or any clone of it. It silently breaks `qwen-sympy`.
- **Do NOT call `qwen.qwen_math_parser.extract_answer`** for `\boxed{}`
  extraction in projects whose datasets contain physics / olympiad latex
  with single-letter variables. Use `extract_boxed_balanced` instead.
- **Do NOT split projects across two envs** (the previous bucket-A /
  bucket-B partition) unless a future dependency forces it. As of
  2026-05-17 it is no longer necessary.
- **Do NOT use HF `math_verify` from inside any trl-projects code.** The
  upstream OPSD `evaluate_math.py` still imports it; when porting OPSD eval
  into our shared env, swap the import for `extract_boxed_balanced` +
  `grade_answer`.

### 4.3 Watch list (known sharp edges)

- GEOQA contains a handful of Chinese `度` characters where the dataset
  expected `°`. The mllm wrapper's `_UNICODE_FIXUPS` must include
  `("度", "")` alongside the existing `("°", "")` entry.
- `qwen.grade_answer` is **exact, not numeric-tolerant**. `35.99 ≠ 36` even
  when both round to the same integer. This is the correct behavior for
  geometry tasks but be aware of it for noisy benchmarks.
- Tuple equality is paren-vs-bracket sensitive: `(1, 2) ≠ [1, 2]` in
  `grade_answer`. If a dataset uses both forms, normalize at the dataset
  layer, not the grader.

---

## 5. Files and code references

- Shared grader: `projects/<any>/verifiers/qwen/math_grade.py` (identical
  bytes across projects).
- Normalizer (Hendrycks): `projects/<any>/verifiers/qwen/math_normalize.py`.
- R1-V tag wrapper: `projects/mllm-co-grpo-dp/verifiers/math_verify_wrapper.py`.
- Avoid: `projects/<any>/verifiers/qwen/qwen_math_parser.py` (the dead parser
  side of the qwen folder, still imported by `qwen_eval.py` which we don't
  call).
