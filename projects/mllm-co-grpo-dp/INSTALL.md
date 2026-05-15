# mllm-co-grpo-dp env setup (新机 / 切机 必读)

> **READ THIS FIRST**
> 这个项目 **必须用独立 conda env**(默认名 `mllm-cogrpodp`),**不能复用 `marti` env**。
> 原因见 §0。
>
> 流程:**§0 理解为什么独立 env → §1 新机检查 → §2 建 env + 装包(按顺序)→ §3 verify → §4 reference baseline**。

---

## §0. 为什么这个项目必须独立 env(决策依据)

trl-projects 仓库下 5 个早期项目(`grpo / un-grpo-maj / co-grpo / co-grpo-dp / co-opd`)共用 `marti` env,它们用 **MARTI 搬过来的 qwen-sympy verifier**(`projects/co-grpo-dp/verifiers/qwen/`)做 reward 判分。这套 verifier 依赖:

- `latex2sympy2==1.9.1`
- `antlr4-python3-runtime==4.7.2`(被 latex2sympy2 1.9.1 锁死)

`mllm-co-grpo-dp` 严格对齐 R1-V baseline,grader 用 **HuggingFace 官方 `math_verify`**。`math_verify` 拉的是:

- `latex2sympy2_extended`
- `antlr4-python3-runtime>=4.13.2`(ATN v4)

→ **两套 antlr4 不能共存于同一 env**。如果在 `marti` 装 `math_verify`,会把 antlr4 升到 4.13.2,qwen-sympy verifier 立刻报:

```
Exception: Could not deserialize ATN with version 3 (expected 4)
```

co-grpo-dp / co-grpo / un-grpo-maj / grpo 的训练全部当场挂(2026-05-05 已亲测踩坑,见 `co_opd_setup` memory)。

**主流学术做法**:每个论文项目 一套独立 conda env + `requirements.txt`(参 NeurIPS Reproducibility checklist / trl / verl / OpenRLHF / R1-V 仓库)。`marti` 共享是历史包袱,**新项目从这里开始切换**到项目独立 env。

---

## §1. 检查阶段(新机/切机必跑)

```bash
# 硬件 + driver
nvidia-smi | head -5
nvcc --version 2>/dev/null || echo "(no nvcc — 不影响,我们走预编译 flash-attn)"

# Conda + Python
which conda && conda info --envs | head -20
# 看是否已经有 mllm-cogrpodp:
conda env list | grep mllm-cogrpodp || echo "(no mllm-cogrpodp env yet — 跑 §2.1)"
```

**对照决策表**:

| §1 输出 | 决策 |
|---|---|
| 没 conda | 先装 Miniconda |
| 没 mllm-cogrpodp env | 跑 §2.1 创建 |
| 已有但缺包 | 跑 §2.x 缺啥补啥(每步前先 `pip show <pkg>` 验证) |
| 已有 + §3 全过 | 不用动,直接开跑 |

---

## §2. 安装步骤(按顺序,有依赖)

### §2.1 创建 conda env + 装 torch

```bash
conda create -n mllm-cogrpodp python=3.12 -y
conda activate mllm-cogrpodp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

**Why**:
- Python 3.12 跟 `marti` env 一致,避免 ABI / wheel 兼容性意外。
- torch 2.10.0+cu128 跟 `marti` env 一致,flash-attn 预编译 wheel 复用同一文件。
- 如果 driver 是 CUDA 11.x,改 `--index-url .../cu118`(但 vLLM 0.18 主要测 12.x)。

### §2.2 trl + dev/vllm/deepspeed(editable 装本仓库)

```bash
cd /path/to/trl-projects     # 仓库根
pip install -e ".[dev,vllm,deepspeed]"
```

**Why**:
- `-e` editable,改 `trl/` 源码无需重装。
- `[vllm,deepspeed]` 是 colocate generate + ZeRO-2 必需。
- 顺带装 transformers / accelerate / datasets / peft 等。

### §2.3 项目额外依赖(VLM + math grader)

```bash
pip install -r projects/mllm-co-grpo-dp/requirements.txt
```

**装到的关键新增**:
- `math-verify`(HF 官方 grader,同 R1-V baseline)+ `latex2sympy2_extended` + antlr4 4.13.2
- `qwen-vl-utils`(Qwen2.5-VL image preprocessing)
- sympy / regex / pylatexenc / word2number(verifier 辅助)

⚠️ **`requirements.txt` 显式不装 `latex2sympy2==1.9.1`**(跟 `latex2sympy2_extended` 冲突)。

### §2.4 flash-attn 预编译 wheel(关键步,容易踩坑)

所有 trainer 硬编码 `attn_implementation="flash_attention_2"`,不装 flash-attn 会启动 fail。

**别 `pip install flash-attn`** —— 默认从源码编译,30-90 分钟、峰值 8-15 GB RAM(经常 OOM 杀进程)、需要 nvcc。**走预编译 wheel,30 秒搞定**。

跟 `marti` env 完全同款,直接复用:

```bash
# 从 https://github.com/Dao-AILab/flash-attention/releases 下载
# 对应 (torch 2.10, cuda 12, py 3.12, cxx11abi FALSE) 的 wheel:
WHEEL=flash_attn-2.8.3+cu12torch2.10cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL}
pip install ${WHEEL}
```

挑 wheel 4 个维度的逻辑参 `SETUP.md §2.4`(同仓库根),不重复。

**装完立刻 verify ABI**:
```bash
python -c "import flash_attn; from flash_attn import flash_attn_func; print('OK', flash_attn.__version__)"
```
报 `undefined symbol` → 99% 是 wheel 跟 torch 主.次不匹配,卸了重挑。

### §2.4½ vllm + transformers 升级解 Gemma-4 锁(**Gemma-4 路径必跑**)

trl 仓库 `pyproject.toml` 把 `[vllm]` extra 锁在 `vllm>=0.11.0,<=0.17.1`,实际装下来:

- vllm 0.17.1 → 要求 `transformers<5`
- transformers 4.57.6(4.x 最高)→ **不认 `model_type: gemma4`**,加载 Gemma-4-E4B-it 报 `KeyError: 'gemma4'`

vllm 各版本 transformers 约束(2026-05-15 实测):

| vllm | transformers 约束 | torch | 备注 |
|---|---|---|---|
| 0.17.1(trl pin) | `<5,>=4.56.0` | 2.10 | 默认装下来的,**不能上 transformers 5.x** |
| 0.18.x | `<5,>=4.56.0` | 2.10 | 同 0.17.x 局限 |
| **0.19.1** | **`!=5.0-5.5, >=4.56`** | **2.10** | **sweet spot — 支持 5.x 且不强升 torch** |
| 0.20+ | 同 0.19.1 | 2.11 | 强升 torch 2.11,放弃,要 cu13 toolkit |

要跑 Gemma-4(Phase 4 cross-sup)**必须**:

```bash
conda activate mllm-cogrpodp
pip install --no-cache-dir vllm==0.19.1                         # 脱离 trl pin
pip install --upgrade --no-cache-dir transformers               # 5.8.1+ for gemma4 support
```

**只跑 Phase 3 (Qwen2.5-VL single)** 不需要这一步 — Qwen2.5-VL 在 transformers 4.57.6 已支持。

**verify**:

```bash
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('google/gemma-4-E4B-it'); print('model_type:', c.model_type)"
# 期望输出: model_type: gemma4
```

### §2.5 HF token + 国内网络(可选)

```bash
huggingface-cli login                            # 粘 token,Gemma-4 是 gated
export HF_ENDPOINT=https://hf-mirror.com         # 国内才需要
```

`google/gemma-4-E4B-it` 是 gated repo,**必须 huggingface 账号同意 license + 装 token**。Qwen2.5-VL-3B-Instruct 不 gated。

⚠️ **共享机器,绝对不要把 token 写进 `.git/config`**(memory: `feedback_no_token_in_git_config`)。push 时别用 `-u`。

---

## §3. Verify(全部过算成功)

```bash
conda activate mllm-cogrpodp
cd /path/to/trl-projects

# 1. torch + cuda
python -c "import torch; print('torch', torch.__version__, 'cuda OK:', torch.cuda.is_available())"

# 2. trl import(editable)
python -c "import trl; print('trl', trl.__version__)"

# 3. transformers 能 load VLM processor
python -c "from transformers import AutoProcessor; print('AutoProcessor OK')"

# 4. flash-attn 真能 forward(要 GPU)
python -c "
import torch
from flash_attn import flash_attn_func
q = k = v = torch.randn(1, 1, 1, 64, dtype=torch.float16, device='cuda')
print('flash-attn forward OK:', flash_attn_func(q, k, v).shape)
"

# 5. math_verify 能 grade(本项目 grader 替代品)
python -c "
from math_verify import parse, verify
gold = parse('\\\\frac{1}{2}')
pred = parse('0.5')
print('math_verify OK:', verify(gold, pred))   # 应该 True
"

# 6. qwen-vl-utils 能 import
python -c "from qwen_vl_utils import process_vision_info; print('qwen-vl-utils OK')"

# 7. GEOQA-style GT self-match(unit + ° 归一化)
python -c "
from math_verify import parse, verify
for gt in ['140', '45°', '\\\\frac{1}{2}', '\\\\sqrt{2}', '2.4cm', '\\\\pi']:
    g = parse(gt); p = parse(gt)
    ok = verify(g, p)
    print(f'self-match {gt!r}: {ok}')
"
# 实测 math_verify 0.9.0 结果(2026-05-15):
#   '140' / '45°' / '\\frac{1}{2}' / '2.4cm' → True  (4 个 ✓)
#   '\\sqrt{2}' / '\\pi'                    → False (2 个 ✗ — math_verify 行为)
# wrapper(`verifiers/math_verify_wrapper.py`)对 empty-parse 有 case-insensitive
# 字符串 fallback,reward 路径上 `grade_answer('\\pi', '\\pi') = True` 仍然成立。
# 但跨 format 的 sqrt/pi 比较仍可能漏(`grade_answer('1.414', '\\sqrt{2}') = False`),
# Phase 1 跑 GEOQA 时若答案里大量 √/π 符号,需要更宽 grader 或额外 normalize。
```

---

## §4. Reference baseline(本机器版本)

新机硬件 / OS 接近时可直接对齐:

| 项 | 值 | 跟 `marti` env 是否一致 |
|---|---|---|
| conda env name | `mllm-cogrpodp` | (独立) |
| Python | 3.12 | ✅ 一致 |
| torch | 2.10.0+cu128 | ✅ |
| torchvision | 0.25.0+cu128 | ✅ |
| CUDA driver (nvidia-smi) | 12.8 | ✅ |
| flash_attn | 2.8.3 (`cu12torch2.10cxx11abiFALSE-cp312`) | ✅ |
| transformers | 4.57.6+(确认能识别 `google/gemma-4-E4B-it`) | ✅(marti 是 4.57.6) |
| trl | editable 装在本仓库 | ✅ |
| accelerate | 1.13.0 | ✅ |
| deepspeed | 0.19.0(实际,2026-05-15) | ✅ |
| **vllm** | **0.19.1**(跑完 §2.4½ 后的版本;装 trl 时默认是 0.17.1) | ❌ marti env 不装 vllm |
| **transformers** | **5.8.1**(2026-05-15 latest;Gemma-4 需要 5.x) | ❌(marti 是 4.57.6,但 marti 不需要 Gemma-4)|
| sympy | 1.14.0 | ✅ |
| **antlr4-python3-runtime** | **4.13.2**(latex2sympy2_extended) | ❌ **marti 是 4.7.2** |
| **latex2sympy2** | **不装**(用 latex2sympy2_extended 替代) | ❌ marti 装的是 1.9.1 |
| **math-verify** | **装**(0.9.0,latest stable) | ❌ marti **不装** |
| **qwen-vl-utils** | **装**(0.0.14,latest stable) | ❌ marti 不需要 |

(以本仓库现机器 `pip show <pkg>` 输出为准。若 transformers 不识别 Gemma-4-E4B-it,见 §2.4½。)

---

## §5. Troubleshooting

| 症状 | 原因 | 解决 |
|---|---|---|
| `Exception: Could not deserialize ATN with version 3 (expected 4)` | 误在 `marti` env 装了 math-verify,或装错 env | 切到 `mllm-cogrpodp` env;`marti` 应保持 antlr4==4.7.2 |
| `pip install flash-attn` 卡几十分钟 | 在编译源码 | Ctrl-C,改 §2.4 预编译 wheel |
| `ImportError: undefined symbol` (flash_attn) | wheel 跟 torch 主.次 不匹配 | `pip uninstall flash-attn -y`,重做 §2.4 |
| `OSError: gated repo (gemma-4-E4B-it)` | 没 HF token 或没 accept license | §2.5;到 HuggingFace 页面点 "Agree and access" |
| `KeyError: 'gemma4'` / `Unrecognized configuration class for Gemma` | transformers 4.x 不认 Gemma-4 | 跑 §2.4½:升 vllm 0.19.1 + transformers 5.x |
| `vllm 0.17.1 requires transformers<5` (pip resolver conflict) | trl pin vllm≤0.17.1 → vllm 0.17 锁 transformers<5 | 同上 §2.4½:vllm 0.19.1 解 transformers<5 锁 |
| `ModuleNotFoundError: math_verify` | §2.3 没跑 | `pip install -r projects/mllm-co-grpo-dp/requirements.txt` |
| `ModuleNotFoundError: qwen_vl_utils` | 同上 | 同上 |
| HF 下载慢/失败 | 国内网络 | §2.5 设镜像 |

---

## §6. 给 Claude 的硬规则(本项目)

1. **绝对不要切到 `marti` env 操作本项目** —— 任何 `pip install` / `pip uninstall` 都只在 `mllm-cogrpodp` env 内做。
2. **绝对不要把 `math-verify` 装进 `marti` env** —— 会破坏 co-grpo-dp 等 4 个项目的 verifier。
3. **flash-attn 永远走预编译**,绝不 `pip install flash-attn`。
4. **首次进仓库先扫**:`~/.claude/projects/-home-yubian/memory/MEMORY.md`(尤其 `mllm_co_grpo_dp_plan` / `co_opd_setup`)。
5. **检查先于安装**:每个 §2.x 进去之前先 `pip show <pkg>` 验证。装了就跳。
6. **不许擅自 commit / push**(用户全局规则);push 时不许 `-u`。
7. **跨 env 切换提醒**:看到 shell prompt 是 `(marti)` 但要装 mllm 包,**立即 stop**,提醒用户:"先 `conda activate mllm-cogrpodp` 再装"。

---

## §7. 跟仓库根 `SETUP.md` 的关系

仓库根 `SETUP.md` 是 5 个老项目(grpo / un-grpo-maj / co-grpo / co-grpo-dp / co-opd)的共享 `marti` env setup,本文件是 `mllm-co-grpo-dp` **新的独立 env** 的 setup。两套并行存在,服务不同 env。

| 文件 | env name | 服务项目 | grader |
|---|---|---|---|
| `SETUP.md`(repo root) | `marti` | 5 个老项目 | qwen-sympy verifier (`co-grpo-dp/verifiers/qwen/`) |
| `projects/mllm-co-grpo-dp/INSTALL.md`(本文件) | `mllm-cogrpodp` | `mllm-co-grpo-dp/` | `math_verify`(HF 官方) |

未来若有第 3 个新项目要再换 grader / 不同 modality / 不同 trl 版本,**继续这个模式**:新建 conda env + 项目目录下自己的 INSTALL.md。
