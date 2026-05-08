# 新机环境配置 (给那边的 Claude 看)

> **READ THIS FIRST**
> 不要看到 `pip install ...` 就直接跑。**每装一个包之前先 §1 检查它是否已经装了 / 装的版本对不对**。装错重装 flash-attn 一次浪费 30+ 分钟,装错 wheel 还会 silent 报 `undefined symbol`。
>
> 流程:**§1 检查 → 把输出贴出来对照 §4 reference → 看 §2 还差哪几步 → 按顺序装 → §3 verify**。

仓库:TRL fork,`https://github.com/williamium3000/trl-projects`,5 个项目共享同一套依赖(`projects/{grpo,un-grpo-maj,co-grpo,co-grpo-dp,co-opd}/`)。

---

## §1. 检查阶段 (必先跑,别跳)

```bash
# 硬件 + driver
nvidia-smi | head -5
nvcc --version 2>/dev/null || echo "(no nvcc — 不影响,我们走预编译 flash-attn)"

# Python + env
python --version
which python
which conda && conda info --envs

# 已装的关键包
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || echo "(no torch)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null || echo "(no flash_attn)"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "(no vllm)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)" 2>/dev/null || echo "(no deepspeed)"
pip show trl 2>/dev/null | head -3 || echo "(no trl)"
```

**对照决策表**:

| §1 输出 | 决策 |
|---|---|
| 没 torch | 必跑 §2.1 |
| 有 torch + cuda 12.x | 跳 §2.1 |
| 没 trl(或 trl 不是 editable 装在本仓库) | 跑 §2.2 |
| 没 sympy / word2number | 跑 §2.3 |
| 没 flash_attn | 必跑 §2.4(**别用 `pip install flash-attn`**) |
| 装的 flash_attn 跟 torch.major.minor 不匹配 | 卸了重装,§2.4 |

---

## §2. 安装步骤 (按顺序,有依赖)

### §2.1 Conda env + torch (没 torch 时)

```bash
conda create -n trl python=3.12 -y
conda activate trl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Why**:
- conda env 隔离 py 版本,不污染系统。
- torch 必须先装,因为 §2.4 的 flash-attn wheel **文件名锁死 `(torch主.次, cuda主, py主.次, cxx11abi)` 4 维**,torch 不存在就挑不了 wheel。
- cu128 对应 CUDA 12.8 driver。如果 §1 看到 `nvidia-smi` driver 是 11.x,改 `--index-url .../cu118`(但 vLLM 主要测试 12.x,尽量用 12)。

### §2.2 trl + dev/vllm/deepspeed

```bash
cd /path/to/trl-projects
pip install -e ".[dev,vllm,deepspeed]"
```

**Why**:
- `-e` editable,改 `trl/` 源码无需重装。
- `[dev,vllm,deepspeed]` 是 `pyproject.toml` 定义的 optional deps;`vllm` 是 colocate generate 必需,`deepspeed` 是 ZeRO-2 + CPU offload 必需。
- 这一步会顺带装 transformers / accelerate / datasets / peft 等。

### §2.3 co-grpo-dp 额外依赖 (verifier)

```bash
pip install -r projects/co-grpo-dp/requirements.txt
```

**Why**:
- verifier(`projects/co-grpo-dp/verifiers/qwen/`,从 MARTI 搬过来)做 latex math 等价判断,需要 `sympy regex latex2sympy2 pylatexenc word2number`。
- `word2number` 缺了 trainer 启动直接 `ImportError`(`qwen_math_parser.py:13`)。

⚠️ **不要 `pip install math-verify`** —— 跟项目自带的 qwen verifier 冲突。

### §2.4 flash-attn 预编译 wheel (关键步,容易踩坑)

所有 trainer 硬编码 `attn_implementation="flash_attention_2"`(`projects/co-grpo-dp/train_co_grpo_dp.py:228` 等),不装会启动 fail。

**别 `pip install flash-attn`** —— 默认从源码编译,30-90 分钟、峰值 8-15 GB RAM(经常 OOM 杀进程)、需要 nvcc。我们用预编译 wheel,30 秒搞定。

**步骤**:

1. 从 §1 拿到 4 个值:torch `X.Y`、cuda `1Z`、Python `A.B`、cxx11abi。
2. 打开 https://github.com/Dao-AILab/flash-attention/releases ,找 latest release。
3. wheel 文件名格式:
   ```
   flash_attn-{VER}+cu{CUDA_MAJOR}torch{TORCH_MAJOR.MINOR}cxx11abi{TRUE|FALSE}-cp{PY_TIGHT}-cp{PY_TIGHT}-linux_x86_64.whl
   ```
   挑选规则:
   - `cu12` ← `torch.version.cuda` 是 `12.x`(`cu118` 给 11.x)
   - `torch2.10` ← `torch.__version__` 是 `2.10.x`(只看主次,不看 patch)
   - `cp312` ← Python 3.12(cp310/cp311/cp312/cp313)
   - `cxx11abiFALSE` ← **conda torch / pip torch 都选 FALSE**;TRUE 只给 PyTorch nightly main build,99% 用不到
4. 装:
   ```bash
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v<VER>/<filename>.whl
   pip install <filename>.whl
   ```

**Why 走预编译**:
| 项 | 源码编译 | 预编译 wheel |
|---|---|---|
| 时间 | 30-90 分钟 | 30 秒 |
| 峰值 RAM | 8-15 GB | 0 |
| 需要 nvcc | 是 | 否 |
| 需要 ninja / cmake | 是 | 否 |

**装完立刻 verify ABI**:
```bash
python -c "import flash_attn; from flash_attn import flash_attn_func; print('OK', flash_attn.__version__)"
```
报 `undefined symbol` / `ImportError` → 99% 是 wheel 跟 torch 主.次 不匹配,卸了重挑 wheel(`pip uninstall flash-attn -y` 然后重来 §2.4)。

### §2.5 HF token (只 Llama gated 才需要)

```bash
huggingface-cli login
```
token 在 https://huggingface.co/settings/tokens ,粘进去。

**Why**:`meta-llama/Llama-3.2-3B-Instruct` 是 gated,首次下载要授权。Qwen 系列不需要。

⚠️ **共享机器,绝对不要把 token 写进 `.git/config`**。push 时别用 `git push -u origin <branch>`(`-u` 会保存 remote 凭据),用 `git push origin <branch>` 就行。push 完 `cat .git/config` 检查没 token。

### §2.6 国内网络 (可选)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## §3. Verify (全部过算成功)

```bash
cd /path/to/trl-projects

# 1. trl import
python -c "import trl; print('trl', trl.__version__)"

# 2. flash-attn 真能 forward,不止能 import(要在 GPU 上)
python -c "
import torch
from flash_attn import flash_attn_func
q = k = v = torch.randn(1, 1, 1, 64, dtype=torch.float16, device='cuda')
print('flash-attn forward OK:', flash_attn_func(q, k, v).shape)
"

# 3. verifier(co-grpo-dp 必需)
python -c "
import sys; sys.path.insert(0, 'projects/co-grpo-dp')
from co_label_utils import grade_answer
print('verifier OK:', grade_answer('1/2', '\\\\frac{1}{2}'))
" 2>/dev/null
# SyntaxWarning 是 vendored qwen 代码自带,无害

# 4. CPU 测试(应 32/32 过)
python3 -W ignore::SyntaxWarning -m pytest projects/co-grpo-dp/tests/ -v
```

---

## §4. Reference baseline (现存机器版本)

新机硬件 / OS 接近时可直接对齐:

| 项 | 值 |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0+cu128 |
| CUDA driver (nvidia-smi) | 12.8 |
| flash_attn | 2.8.3 (`cu12torch2.10cxx11abiFALSE-cp312`) |
| trl | editable 装在仓库内 |

(确认当前以本仓库现机器 `pip show <pkg>` 输出为准。)

---

## §5. Troubleshooting

| 症状 | 原因 | 解决 |
|---|---|---|
| `pip install flash-attn` 卡几十分钟 | 在编译源码 | Ctrl-C,改用 §2.4 预编译 wheel |
| `ImportError: undefined symbol` (flash_attn) | wheel 跟 torch 主.次不匹配 | `pip uninstall flash-attn -y`,重做 §2.4 |
| `ModuleNotFoundError: word2number` 启动 trainer | §2.3 没跑 | `pip install -r projects/co-grpo-dp/requirements.txt` |
| `OSError: gated repo` (Llama) | 没 HF token | §2.5 |
| `RuntimeError: CUDA error: no kernel image` | torch 跟 driver 不兼容(driver 太旧) | `nvidia-smi` 看 driver 版本,把 torch 降到 cu118 |
| HF 下载慢/失败 | 国内网络 | §2.6 设镜像 |

更多 runtime 错误(rendezvous hang / vLLM OOM 等)看 `projects/co-grpo-dp/README.md` §10。

---

## §6. 给 Claude 的硬规则

1. **检查先于安装**:每个 §2.x 进去之前先 `pip show <pkg>` / `python -c "import <pkg>"` 验证。装了就跳,装错就先卸。
2. **flash-attn 永远走预编译**,绝不 `pip install flash-attn` 让它编源码。
3. **memory 里有项目 context**,首次进仓库先扫 `~/.claude/projects/-home-yubian/memory/MEMORY.md`。
4. **仓库内规约**:`./CLAUDE.md`(repo 根)+ `projects/co-grpo-dp/CLAUDE.md`(子项目)。
5. **不许擅自 commit / push**(用户全局规则);push 时不许 `-u`(memory: `feedback_no_token_in_git_config`)。
