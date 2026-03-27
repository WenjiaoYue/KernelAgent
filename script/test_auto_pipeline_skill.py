"""
OpenClaw Agent Pipeline - auto_run + auto_quant + auto_eval (Skill-based)

Runs all three tasks sequentially for a single model.
Each task uses its own session key so transcripts are kept separate.
The shared workspace_dir (output of auto_run) is passed to all tasks
so auto_quant and auto_eval can reuse the venv via model_info.json.

Usage:
    python3 test_auto_pipeline_skill.py

    # Override via env vars:
    MODEL_ID=Qwen/Qwen3-1.7B DEVICE=xpu python3 test_auto_pipeline_skill.py

    # Run only specific stages (comma-separated):
    STAGES=auto_run,auto_eval python3 test_auto_pipeline_skill.py
"""

import subprocess
import json
import os
import shutil
import sys
from pathlib import Path


# =============================================================================
# Helpers (shared by all stages)
# =============================================================================

def get_session_id_from_key(session_key: str) -> str:
    try:
        with open("/root/.openclaw/agents/main/sessions/sessions.json", "r") as f:
            data = json.load(f)
        for key in [f"agent:main:{session_key}", session_key]:
            if key in data:
                return data[key].get("sessionId", session_key)
        return session_key
    except Exception:
        return session_key


def run_agent_task(task: str, session_key: str, timeout: int,
                   transcript_dest: str = None) -> dict:
    cmd = [
        "openclaw", "agent",
        "--session-id", session_key,
        "--message", task,
        "--deliver",
        "--timeout", str(timeout)
    ]
    print(f"  ⏱️  timeout: {timeout}s  session_key: {session_key}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    session_id = get_session_id_from_key(session_key)
    transcript_path = f"/root/.openclaw/agents/main/sessions/{session_id}.jsonl"

    copied_transcript = None
    if transcript_dest:
        Path(transcript_dest).mkdir(parents=True, exist_ok=True)
        dest = Path(transcript_dest) / "transcript.jsonl"
        try:
            shutil.copy2(transcript_path, dest)
            copied_transcript = str(dest)
        except Exception as e:
            print(f"  ⚠️  Failed to copy transcript: {e}")

    return {
        "session_key": session_key,
        "session_id": session_id,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "transcript_path": transcript_path,
        "copied_transcript": copied_transcript,
    }


def print_result(stage: str, result: dict):
    status = "✅ OK" if result["returncode"] == 0 else "❌ FAILED"
    print(f"\n[{stage}] {status}")
    print(f"  session_key:        {result['session_key']}")
    if result.get('copied_transcript'):
        print(f"  transcript (已存入workspace): {result['copied_transcript']}")
    else:
        print(f"  transcript (uuid路径):   {result['transcript_path']}")
    if result.get("stderr"):
        print(f"  stderr:\n{result['stderr'][:500]}")


# =============================================================================
# Task builders
# =============================================================================

def build_auto_run_task(model_id, workspace_dir, device, skill_path):
    return f"""
You are an expert in LLM inference. Your task is to run a quick smoke test on a large language model following best practices.
You need use this skill: {skill_path}

Please test inference of the model {model_id}.
Device: {device}
Output directory: {workspace_dir}
HuggingFace cache: /storage/lkk/cache

Generate at most 50 tokens. Save logs and summary.md to the output directory.
After environment setup, write model_info.json to {workspace_dir} so that subsequent tasks (auto_quant, auto_eval) can reuse the venv without recreating it.
"""


def build_auto_quant_task(model_id, workspace_dir, output_dir, scheme, method, export_format, device, skill_path):
    return f"""
You are an expert in LLM quantization using the Intel Auto-Round toolkit. Your task is to quantize a large language model following best practices.
You need use this skill for quantization: {skill_path}

Please quantize {model_id} into {scheme} format with {method} method, and export '{export_format}' format.
The output dir is {output_dir}.
The runtime device is on {device}, you should run the quantization code on {device} device.

IMPORTANT: Before creating a new venv, check if {workspace_dir}/model_info.json exists.
If it exists, read venv_path from it and reuse that venv (just pip install auto-round into it).
This avoids recreating the environment that auto_run already set up.
"""


def build_auto_eval_task(model_path, workspace_dir, tasks, output_path, device, batch_size, max_model_len, skill_path):
    return f"""
You are an expert in LLM evaluation using vLLM and lm-evaluation-harness. Your task is to evaluate a quantized large language model following best practices.
You need use this skill for evaluation: {skill_path}

Please evaluate the model at {model_path}.
Evaluation tasks: {tasks}
Output directory: {output_path}
Device: {device}
Batch size: {batch_size}
Max model length: {max_model_len}

IMPORTANT: Before installing anything, check if {workspace_dir}/model_info.json exists.
If it exists, read venv_path from it and reuse that venv (just pip install lm_eval vllm into it).
This avoids recreating the environment that auto_run already set up.
"""


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Auto Pipeline: auto_run → auto_quant → auto_eval")
    print("=" * 60)

    # ===== Settings (all overridable via env vars) =====
    TIMEOUT        = int(os.getenv("TIMEOUT",        "7200"))
    MODEL_ID       = os.getenv("MODEL_ID",           "Qwen/Qwen3-0.6B")
    DEVICE         = os.getenv("DEVICE",             "xpu")

    # Stages to run (default: all three)
    # Example: STAGES=auto_run,auto_quant  to skip auto_eval
    STAGES         = os.getenv("STAGES", "auto_run,auto_quant,auto_eval").split(",")

    # Shared workspace — auto_run writes model_info.json here
    _model_slug    = MODEL_ID.replace("/", "_")
    WORKSPACE_DIR  = os.getenv("WORKSPACE_DIR", f"/storage/lkk/inference/{_model_slug}")

    # auto_quant settings
    SCHEME         = os.getenv("SCHEME",         "W4A16")
    METHOD         = os.getenv("METHOD",         "RTN")
    EXPORT_FORMAT  = os.getenv("EXPORT_FORMAT",  "auto_round:auto_gptq")
    QUANT_OUT_DIR  = os.getenv("QUANT_OUT_DIR",  f"/kaokao/quantized/{_model_slug}-{SCHEME}")

    # auto_eval settings — evaluates the quantized model by default
    EVAL_MODEL     = os.getenv("EVAL_MODEL",     QUANT_OUT_DIR)
    EVAL_TASKS     = os.getenv("EVAL_TASKS",     "piqa")
    EVAL_OUT_DIR   = os.getenv("EVAL_OUT_DIR",   f"/kaokao/lm_eval_results/{_model_slug}")
    BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "8"))
    MAX_MODEL_LEN  = int(os.getenv("MAX_MODEL_LEN", "8192"))

    # Skill paths (inside container)
    AUTO_RUN_SKILL   = os.getenv("AUTO_RUN_SKILL",   "/root/.openclaw/workspace/skills/auto_run/SKILL.md")
    AUTO_QUANT_SKILL = os.getenv("AUTO_QUANT_SKILL", "/root/.openclaw/workspace/skills/auto_quant/SKILL.md")
    AUTO_EVAL_SKILL  = os.getenv("AUTO_EVAL_SKILL",  "/root/.openclaw/workspace/skills/auto_eval/SKILL.md")

    print(f"\nModel:           {MODEL_ID}")
    print(f"Device:          {DEVICE}")
    print(f"Stages:          {', '.join(STAGES)}")
    print(f"Workspace dir:   {WORKSPACE_DIR}  (shared: venv + model_info.json)")
    print(f"Quant output:    {QUANT_OUT_DIR}")
    print(f"Eval model:      {EVAL_MODEL}")
    print(f"Eval tasks:      {EVAL_TASKS}")
    print()

    results = {}

    # --- Stage 1: auto_run ---
    if "auto_run" in STAGES:
        print("\n" + "─" * 60)
        print("Stage 1/3: auto_run (inference smoke test)")
        print("─" * 60)
        session_key = f"pipeline_{_model_slug}_autorun"
        task = build_auto_run_task(MODEL_ID, WORKSPACE_DIR, DEVICE, AUTO_RUN_SKILL)
        results["auto_run"] = run_agent_task(task, session_key=session_key, timeout=TIMEOUT,
                                              transcript_dest=f"{WORKSPACE_DIR}/auto_run")
        print_result("auto_run", results["auto_run"])

    # --- Stage 2: auto_quant ---
    if "auto_quant" in STAGES:
        print("\n" + "─" * 60)
        print("Stage 2/3: auto_quant (quantization)")
        print("─" * 60)
        session_key = f"pipeline_{_model_slug}_autoquant"
        task = build_auto_quant_task(
            MODEL_ID, WORKSPACE_DIR, QUANT_OUT_DIR,
            SCHEME, METHOD, EXPORT_FORMAT, DEVICE, AUTO_QUANT_SKILL
        )
        results["auto_quant"] = run_agent_task(task, session_key=session_key, timeout=TIMEOUT,
                                               transcript_dest=f"{WORKSPACE_DIR}/auto_quant")
        print_result("auto_quant", results["auto_quant"])

    # --- Stage 3: auto_eval ---
    if "auto_eval" in STAGES:
        print("\n" + "─" * 60)
        print("Stage 3/3: auto_eval (evaluation)")
        print("─" * 60)
        session_key = f"pipeline_{_model_slug}_autoeval"
        task = build_auto_eval_task(
            EVAL_MODEL, WORKSPACE_DIR, EVAL_TASKS, EVAL_OUT_DIR,
            DEVICE, BATCH_SIZE, MAX_MODEL_LEN, AUTO_EVAL_SKILL
        )
        results["auto_eval"] = run_agent_task(task, session_key=session_key, timeout=TIMEOUT,
                                              transcript_dest=f"{WORKSPACE_DIR}/auto_eval")
        print_result("auto_eval", results["auto_eval"])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"\nWorkspace dir: {WORKSPACE_DIR}")
    print(f"  ├── auto_run/transcript.jsonl")
    print(f"  ├── auto_quant/transcript.jsonl")
    print(f"  └── auto_eval/transcript.jsonl")
    print()
    for stage, r in results.items():
        status = "✅" if r["returncode"] == 0 else "❌"
        loc = r.get('copied_transcript') or r['transcript_path']
        print(f"  {status} {stage:12s}  {loc}")
    print()
