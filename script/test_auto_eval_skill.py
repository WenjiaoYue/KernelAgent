"""
OpenClaw Agent Task - Auto-Eval (Skill-based)

Usage:
    python3 test_auto_eval_skill.py

    # Override via env vars:
    MODEL_PATH=/kaokao/quantized/Qwen3-0.6B-W4A16 TASKS=piqa,hellaswag python3 test_auto_eval_skill.py

使用 openclaw agent 发送任务，agent 读取 auto_eval SKILL.md 执行模型评测。
"""

import subprocess
import json
import os
import shutil
from pathlib import Path


def get_active_session_key():
    """获取当前活跃的 session key"""
    result = subprocess.run(
        ["openclaw", "sessions", "--json"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        sessions = data.get("sessions", [])
        if sessions:
            return sessions[0].get("sessionKey", "main")
    return "main"


def get_session_id_from_key(session_key: str) -> str:
    """根据 sessionKey 查找对应的 sessionId"""
    try:
        with open("/root/.openclaw/agents/main/sessions/sessions.json", "r") as f:
            data = json.load(f)

        key_variants = [f"agent:main:{session_key}", session_key]

        for key in key_variants:
            if key in data:
                return data[key].get("sessionId", session_key)

        return session_key
    except Exception:
        return session_key


def run_agent_task(task: str, session_key: str = None, timeout: int = 7200,
                   transcript_dest: str = None) -> dict:
    """
    通过 OpenClaw CLI 运行 agent 任务

    Args:
        task: 给 agent 的指令
        session_key: sessionKey (如 "main", "my_task")，不是 UUID!
        timeout: 超时时间(秒)，评测任务建议 2 小时以上
        transcript_dest: 任务结束后把 transcript 复制到这个目录(如 workspace_dir/auto_eval/)
    """
    if not session_key:
        session_key = get_active_session_key()

    cmd = [
        "openclaw", "agent",
        "--session-id", session_key,
        "--message", task,
        "--deliver",
        "--timeout", str(timeout)
    ]

    print(f"⏱️  Agent timeout: {timeout} 秒 ({timeout/60:.0f} 分钟)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 60
    )

    session_id = get_session_id_from_key(session_key)
    transcript_path = f"/root/.openclaw/agents/main/sessions/{session_id}.jsonl"

    # 把 transcript 复制到 model workspace 目录，方便按模型查找
    copied_transcript = None
    if transcript_dest:
        Path(transcript_dest).mkdir(parents=True, exist_ok=True)
        dest = Path(transcript_dest) / "transcript.jsonl"
        try:
            shutil.copy2(transcript_path, dest)
            copied_transcript = str(dest)
            print(f"📄 Transcript copied → {dest}")
        except Exception as e:
            print(f"⚠️  Failed to copy transcript: {e}")

    return {
        "session_key": session_key,
        "session_id": session_id,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "transcript_path": transcript_path,
        "copied_transcript": copied_transcript,
    }


# =============================================================================
# 任务 - 使用 auto_eval skill 评测模型
# =============================================================================

def build_task(
    model_path: str,
    workspace_dir: str,
    tasks: str,
    output_path: str,
    device: str,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    skill_path: str,
) -> str:
    return f"""
You are an expert in LLM evaluation using vLLM and lm-evaluation-harness. Your task is to evaluate a quantized large language model following best practices.
You need use this skill for evaluation: {skill_path}

Please evaluate the model at {model_path}.
Evaluation tasks: {tasks}
Output directory: {output_path}
Device: {device}
Batch size: {batch_size}
Max model length: {max_model_len}
GPU memory utilization: {gpu_memory_utilization}

IMPORTANT: Before installing anything, check if {workspace_dir}/model_info.json exists.
If it exists, read venv_path from it and reuse that venv (just pip install lm_eval vllm into it).
This avoids recreating the environment that auto_run already set up.

Follow the skill instructions to detect the quantization format, configure vLLM args accordingly, and run lm-eval.
"""


if __name__ == "__main__":

    print("=" * 60)
    print("Auto-Eval Test - Skill 方式")
    print("=" * 60)

    # ===== 关键设置（可通过环境变量覆盖）=====
    TIMEOUT                 = int(os.getenv("TIMEOUT",                  "7200"))

    MODEL_ID                = os.getenv("MODEL_ID",                     "Qwen/Qwen3-0.6B")
    SCHEME                  = os.getenv("SCHEME",                       "W4A16")  # 与 auto_quant 保持一致
    TASKS                   = os.getenv("TASKS",                        "piqa")
    DEVICE                  = os.getenv("DEVICE",                       "xpu")
    BATCH_SIZE              = int(os.getenv("BATCH_SIZE",               "8"))
    MAX_MODEL_LEN           = int(os.getenv("MAX_MODEL_LEN",            "8192"))
    GPU_MEMORY_UTILIZATION  = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
    SKILL_PATH              = os.getenv("SKILL_PATH",                   "/root/.openclaw/workspace/skills/auto_eval/SKILL.md")

    # Workspace dir — use MODEL_ID slug (same as auto_run / auto_quant)
    _model_slug    = MODEL_ID.replace("/", "_")
    WORKSPACE_DIR           = os.getenv("WORKSPACE_DIR",  f"/storage/lkk/inference/{_model_slug}")

    # 输入：auto_quant 的输出路径，与 pipeline 保持一致
    MODEL_PATH              = os.getenv("MODEL_PATH",     f"/kaokao/quantized/{_model_slug}-{SCHEME}")
    # 输出：lm_eval 结果目录
    OUTPUT_PATH             = os.getenv("OUTPUT_PATH",    f"/kaokao/lm_eval_results/{_model_slug}")

    # SESSION_KEY 从 MODEL_ID 派生，与 backend.py 的 _session_key_for() 保持一致
    # backend 期望：autoeval_{MODEL_ID.replace("/","_")}，例如 autoeval_Qwen_Qwen3-0.6B
    _default_session_key = "autoeval_" + _model_slug
    SESSION_KEY             = os.getenv("SESSION_KEY",    _default_session_key)

    print(f"\n模型 ID:          {MODEL_ID}")
    print(f"量化方案:         {SCHEME}")
    print(f"模型路径(输入):    {MODEL_PATH}")
    print(f"评测任务:         {TASKS}")
    print(f"运行设备:         {DEVICE}")
    print(f"输出目录:         {OUTPUT_PATH}")
    print(f"Workspace dir:    {WORKSPACE_DIR}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Max model len:   {MAX_MODEL_LEN}")
    print(f"GPU mem util:    {GPU_MEMORY_UTILIZATION}")
    print(f"Skill 路径:       {SKILL_PATH}")
    print(f"使用 session_key: {SESSION_KEY}")
    print(f"Agent timeout:   {TIMEOUT} 秒 ({TIMEOUT/60:.0f} 分钟)")
    print()

    task = build_task(
        model_path=MODEL_PATH,
        workspace_dir=WORKSPACE_DIR,
        tasks=TASKS,
        output_path=OUTPUT_PATH,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        skill_path=SKILL_PATH,
    )

    print("--- Prompt ---")
    print(task)
    print("-" * 60)

    result = run_agent_task(task, session_key=SESSION_KEY, timeout=TIMEOUT,
                            transcript_dest=f"{WORKSPACE_DIR}/auto_eval")

    print(f"\n--- 结果 ---")
    print(f"传入的 session_key: {result['session_key']}")
    print(f"实际的 session_id:  {result['session_id']}")
    print(f"Transcript 原始位置: {result['transcript_path']}")
    if result.get('copied_transcript'):
        print(f"Transcript 已复制至: {result['copied_transcript']}")
    print(f"Agent returncode:  {result['returncode']}")
    print(f"Agent returncode:  {result['returncode']}")

    if result.get("stderr"):
        print(f"\nStderr:\n{result['stderr']}")

    print("\n" + "=" * 60)
