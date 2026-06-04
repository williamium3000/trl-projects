#!/usr/bin/env python3
"""Evaluate ONE uploaded text-LLM checkpoint on MATH-500 (pass@1).

Faithfully reproduces the training-time inline eval (verified 2026-05-30):
  - data    : data/math500/test.json (the same MATH500_EVAL_PATH the trainer used)
  - prompt  : [{"role":"user","content": f"{q}\\n {INSTRUCTION}"}] -> model chat
              template, add_generation_prompt=True, no system message
  - sampling: temperature=0.6 (temperature_eval), top_p=1.0, top_k disabled,
              max_tokens=3072, n=1  (pass@1)
  - verifier: verifiers.qwen  extract_answer(text,"math") + grade_answer
              (the SAME function objects the trainer's reward_correctness uses)

Usage (normally invoked by run_eval_uploaded_ckpts.sh, one repo per GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_uploaded_ckpts.py <HF_REPO> [N] [OUT_JSON]
        N        : #MATH-500 problems (default 500 = full; use 30 for a quick smoke)
        OUT_JSON : where to write the result (default /tmp/eval_<repo>.json)
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
# verifiers/ lives next to this file (projects/co-grpo-dp/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    repo = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out_json = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/eval_{repo.split('/')[-1]}.json"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from verifiers.qwen.qwen_math_parser import extract_answer
    from verifiers.qwen.math_grade import grade_answer

    # MATH-500 lives at repo-root data/; resolve relative to this file (../../../data)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    data_path = os.path.join(repo_root, "data", "math500", "test.json")
    data = json.load(open(data_path))[:N]

    tok = AutoTokenizer.from_pretrained(repo)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{e['prompt']}\n {INSTRUCTION}"}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in data
    ]

    # Gemma-3 ships config.architectures=Gemma3ForConditionalGeneration (multimodal);
    # vLLM then hunts for an image processor and crashes. Force the text class.
    hf_overrides = {}
    if "gemma" in repo.lower():
        hf_overrides = {"architectures": ["Gemma3ForCausalLM"]}

    llm = LLM(model=repo, dtype="bfloat16", max_model_len=3584,
              gpu_memory_utilization=0.85, trust_remote_code=True,
              hf_overrides=hf_overrides)
    sp = SamplingParams(temperature=0.6, top_p=1.0, max_tokens=3072, n=1)
    outs = llm.generate(prompts, sp)

    correct = 0
    for e, o in zip(data, outs):
        pred = extract_answer(o.outputs[0].text, "math")
        if pred is not None and grade_answer(pred, e["answer"]):
            correct += 1
    acc = correct / len(data)

    print(f"\n==== {repo}  MATH-500[:{N}]  pass@1 = {correct}/{len(data)} = {acc:.4f} ====")
    json.dump({"repo": repo, "n": len(data), "correct": correct, "acc": acc},
              open(out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
