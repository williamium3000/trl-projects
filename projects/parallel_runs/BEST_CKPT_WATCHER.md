# best_ckpt_watcher —— best ckpt 兜底守护(MLLM + LLM 全覆盖)

## 它干什么 / 为什么需要
训练里的原生 `BestKeeperCallback`(`train_mllm_single.py` / `train_co_grpo_dp.py` / LLM trainers)**实时大面积失效**:多数 run 不生成 `best_model`(尤其 InternVL、colearn 两组)。配合 `save_total_limit=3`,**早期的最佳 checkpoint 会被轮转删掉 → 永久丢失**。

这个 watcher 是**外挂兜底**:周期性扫所有 run、按 `eval_reward` 挑出最佳 checkpoint、硬链到 `<run>/best_model/` + 写 `best_metric.json`,并用 `best_model.watcher.json` 记全局 best(**即使源 checkpoint 之后被轮转删了,best_model 仍在**)。不动训练、不重启。

## 覆盖范围(一个实例够了)
- 扫 **两个 repo** 的 `work_dirs`:`trl-projects-mllm`(MLLM)+ `trl-projects/projects`(LLM 7B/3B)。
- 认所有带 `checkpoint-*` 的目录:`_eval_curve` 单模型 run、`phase4` 的 `model_a`/`model_b`、co-grpo 的 `group_A`/`group_B`。
- work_dirs 在**共享 NAS** → **任一存活 pod 跑一个实例,就覆盖全集群所有 pod 的所有 run**。

## 怎么跑(任一存活 pod,只跑一个)
```bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project
nohup python3 best_ckpt_watcher.py > /tmp/best_ckpt_watcher.log 2>&1 &
# 看它在干活:tail -f /tmp/best_ckpt_watcher.log
```
⚠️ pod 被 kill 后,**在别的存活 pod 重新跑一次上面命令**(进程死了,但它写的 best_model 都在 NAS 不丢)。
⚠️ 只跑**一个**实例(多个会互相 rmtree/重建 best_model,无害但浪费)。

## 完整代码(复制即用,存成 `best_ckpt_watcher.py`)
```python
#!/usr/bin/env python3
# 外挂 best-ckpt watcher:周期扫所有 run,按 eval_reward 挑 best checkpoint,
# 硬链到 <run>/best_model + 写 best_metric.json。补 BestKeeperCallback 实时失效的洞,
# 不动训练、不重启。每个 run 维护 best_model.watcher.json 记全局 best(防 checkpoint 轮转丢)。
import os, json, glob, shutil, time, sys

ROOTS = [
    "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects-mllm/work_dirs",
    "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs",
]
METRIC = "eval_reward"
INTERVAL = 600  # 10 min

def ckpt_eval(ckpt):
    """该 checkpoint 的 eval_reward(取其 trainer_state.log_history 里最后一个 eval_reward)。"""
    ts = os.path.join(ckpt, "trainer_state.json")
    if not os.path.exists(ts):
        return None
    try:
        st = json.load(open(ts))
    except Exception:
        return None
    v = None
    for e in st.get("log_history", []):
        if METRIC in e:
            v = e[METRIC]
    return v

def has_weights(d):
    return bool(glob.glob(os.path.join(d, "*.safetensors")))

def process_run(run):
    """run = 含 checkpoint-* 的目录(_eval_curve 的 run,或 phase4 的 model_a/model_b)。"""
    ckpts = [c for c in glob.glob(os.path.join(run, "checkpoint-*")) if os.path.isdir(c) and has_weights(c)]
    if not ckpts:
        return
    state_f = os.path.join(run, "best_model.watcher.json")
    prev = {}
    if os.path.exists(state_f):
        try: prev = json.load(open(state_f))
        except Exception: prev = {}
    best_v = prev.get("value", None)
    # 在 *当前存在* 的 checkpoint 里找最高 eval
    cand = []
    for c in ckpts:
        v = ckpt_eval(c)
        if v is not None:
            cand.append((v, int(c.rsplit("-", 1)[1]), c))
    if not cand:
        return
    cand.sort()
    cv, cstep, cdir = cand[-1]  # 当前存在的最佳
    # 若超过历史记录 → 更新 best_model
    if best_v is None or cv > best_v + 1e-9:
        dst = os.path.join(run, "best_model")
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            try:
                shutil.copytree(cdir, dst, copy_function=os.link)   # 同 NAS 0 拷贝
            except Exception:
                shutil.copytree(cdir, dst)                          # 跨设备回退真拷贝
            json.dump({"step": cstep, "metric": METRIC, "value": float(cv)},
                      open(os.path.join(run, "best_metric.json"), "w"), indent=2)
            json.dump({"step": cstep, "value": float(cv)}, open(state_f, "w"))
            print(f"[{time.strftime('%H:%M:%S')}] best↑ {run.split('work_dirs/')[-1]} step={cstep} {METRIC}={cv:.4f}", flush=True)
        except Exception as e:
            print(f"[WARN] {run}: {e}", flush=True)

def scan():
    runs = set()
    for root in ROOTS:
        for d in glob.glob(os.path.join(root, "**", "checkpoint-*"), recursive=True):
            runs.add(os.path.dirname(d))
    for r in sorted(runs):
        process_run(r)

if __name__ == "__main__":
    print(f">>> best_ckpt_watcher 启动,每 {INTERVAL}s 扫一次", flush=True)
    while True:
        try:
            scan()
        except Exception as e:
            print(f"[scan err] {e}", flush=True)
        time.sleep(INTERVAL)
```

## 验证它生效
```bash
# 任一 run 应有 best_model 权重 + best_metric.json
ls <run>/best_model/*.safetensors
cat <run>/best_metric.json     # {"step":..,"metric":"eval_reward","value":..}
# colearn(phase4):看 model_a/best_model 和 model_b/best_model
```
