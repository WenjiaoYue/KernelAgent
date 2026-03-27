"""
OpenClaw Agent Task - Auto-Round Quantization (Skill-based)

Usage:
    python3 test_auto_quant_skill.py

    # Override via env vars:
    MODEL_ID=Qwen/Qwen3-1.7B DEVICE=xpu python3 test_auto_quant_skill.py

使用 openclaw agent 发送任务，agent 读取 auto_quant SKILL.md 执行量化。
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
        timeout: 超时时间(秒)，量化任务建议 2 小时以上
        transcript_dest: 任务结束后把 transcript 复制到这个目录(如 workspace_dir/auto_quant/)
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
# 任务 - 使用 auto_quant skill 量化模型
# =============================================================================

def build_task(
    model_id: str,
    output_dir: str,
    scheme: str,
    method: str,
    export_format: str,
    device: str,
    skill_path: str,
) -> str:
    return f"""
You are an expert in LLM quantization using the Intel Auto-Round toolkit. Your task is to quantize a large language model following best practices.
You need use this skill for quantization: {skill_path}

Please quantize {model_id} into {scheme} format with {method} method, and export '{export_format}' format.
The output dir is {output_dir}.
The runtime device is on {device}, you should run the quantization code on {device} device.
"""


if __name__ == "__main__":

    print("=" * 60)
    print("Auto-Round Quantization Test - Skill 方式")
    print("=" * 60)

    # ===== 关键设置（可通过环境变量覆盖）=====
    TIMEOUT        = int(os.getenv("TIMEOUT",       "7200"))

    MODEL_ID       = os.getenv("MODEL_ID",          "Qwen/Qwen3-0.6B")
    SCHEME         = os.getenv("SCHEME",            "W4A16")
    METHOD         = os.getenv("METHOD",            "RTN")
    EXPORT_FORMAT  = os.getenv("EXPORT_FORMAT",     "auto_round:auto_gptq")
    DEVICE         = os.getenv("DEVICE",            "xpu")
    SKILL_PATH     = os.getenv("SKILL_PATH",        "/wenjiao/openclaw-triton-gen/examples/auto_quant/SKILL.md")

    # Workspace dir (shared with auto_run) — transcript 复制到此处
    _model_slug    = MODEL_ID.replace("/", "_")
    WORKSPACE_DIR  = os.getenv("WORKSPACE_DIR",     f"/storage/lkk/inference/{_model_slug}")

    # 量化输出路径 — 与 pipeline 保持一致：/kaokao/quantized/{model}-{scheme}
    OUTPUT_DIR     = os.getenv("OUTPUT_DIR",        f"/kaokao/quantized/{_model_slug}-{SCHEME}")

    # SESSION_KEY 默认从 MODEL_ID + SCHEME 派生，保证每个任务有独立的 session 文件
    _default_session_key = "autoquant_" + _model_slug + "_" + SCHEME
    SESSION_KEY    = os.getenv("SESSION_KEY",       _default_session_key)

    print(f"\n模型:           {MODEL_ID}")
    print(f"量化方案:        {SCHEME} / {METHOD}")
    print(f"导出格式:        {EXPORT_FORMAT}")
    print(f"运行设备:        {DEVICE}")
    print(f"输出目录:        {OUTPUT_DIR}")
    print(f"Workspace dir:   {WORKSPACE_DIR}")
    print(f"Skill 路径:      {SKILL_PATH}")
    print(f"使用 session_key: {SESSION_KEY}")
    print(f"Agent timeout:   {TIMEOUT} 秒 ({TIMEOUT/60:.0f} 分钟)")
    print()

    task = build_task(
        model_id=MODEL_ID,
        output_dir=OUTPUT_DIR,
        scheme=SCHEME,
        method=METHOD,
        export_format=EXPORT_FORMAT,
        device=DEVICE,
        skill_path=SKILL_PATH,
    )

    print("--- Prompt ---")
    print(task)
    print("-" * 60)

    result = run_agent_task(task, session_key=SESSION_KEY, timeout=TIMEOUT,
                            transcript_dest=f"{WORKSPACE_DIR}/auto_quant")

    print(f"\n--- 结果 ---")
    print(f"传入的 session_key: {result['session_key']}")
    print(f"实际的 session_id:  {result['session_id']}")
    print(f"Transcript 原始位置: {result['transcript_path']}")
    if result.get('copied_transcript'):
        print(f"Transcript 已复制至: {result['copied_transcript']}")
    print(f"Agent returncode:  {result['returncode']}")

    if result.get("stderr"):
        print(f"\nStderr:\n{result['stderr']}")

    print("\n" + "=" * 60)
