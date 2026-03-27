"""
OpenClaw Agent Task - Auto-Run Inference Smoke Test

Usage:
    python3 test_auto_run_skill.py

使用 openclaw agent 发送任务，agent 读取 auto_run SKILL.md 执行推理冒烟测试。
目标：验证模型能否在目标设备上正常加载并生成输出。
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
        timeout: 超时时间(秒)
        transcript_dest: 任务结束后把 transcript 复制到这个目录(如 workspace_dir/auto_run/)
    """
    if not session_key:
        session_key = get_active_session_key()

    # 把 HF_ENDPOINT / HF_HOME 透传给 openclaw 子进程
    env = os.environ.copy()

    cmd = [
        "openclaw", "agent",
        "--session-id", session_key,
        "--message", task,
        "--local",
        "--timeout", str(timeout)
    ]

    print(f"⏱️  Agent timeout: {timeout} 秒 ({timeout/60:.0f} 分钟)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        env=env,
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
# 任务 - 使用 auto_run skill 做推理冒烟测试
# =============================================================================

def build_task(model_id: str, device: str, device_index: str, output_dir: str) -> str:
    hf_cache    = os.getenv("HF_HOME",     "/storage/lkk/cache")
    hf_endpoint = os.getenv("HF_ENDPOINT", "")
    proxy_line  = f"HuggingFace mirror/proxy: {hf_endpoint}" if hf_endpoint else ""
    return f"""
You are an expert in LLM inference. Your task is to run a quick smoke test on a large language model following best practices.
Use the auto_run skill.

Please test inference of the model {model_id}.
Device: {device}:{device_index}
Output directory: {output_dir}
HuggingFace cache: {hf_cache}
{proxy_line}

Generate at most 50 tokens.

IMPORTANT: Save ALL output files (inference_script.py, logs/, summary.md) inside the subdirectory {output_dir}/auto_run/ — NOT in {output_dir} directly.
Final structure must be:
  {output_dir}/auto_run/inference_script.py
  {output_dir}/auto_run/logs/inference.log
  {output_dir}/auto_run/summary.md
"""


if __name__ == "__main__":

    print("=" * 60)
    print("Auto-Run Inference Smoke Test - Agent 方式")
    print("=" * 60)

    # ===== 关键设置（可通过环境变量覆盖）=====
    TIMEOUT       = int(os.getenv("TIMEOUT",   "7200"))

    MODEL_ID      = os.getenv("MODEL_ID",      "Qwen/Qwen3-0.6B")
    DEVICE        = os.getenv("DEVICE",        "cuda")   # cuda | xpu | cpu
    DEVICE_INDEX  = os.getenv("DEVICE_INDEX",  "0")
    OUTPUT_DIR    = os.getenv("OUTPUT_DIR",    f"/storage/lkk/inference/{MODEL_ID.replace('/', '_')}")
    HF_CACHE      = os.getenv("HF_HOME",       "/storage/lkk/cache")
    HF_ENDPOINT   = os.getenv("HF_ENDPOINT",   "")   # HuggingFace 镜像/代理，如 https://hf-mirror.com

    # SESSION_KEY 默认从 MODEL_ID 派生，保证每个模型有独立的 session 文件
    # 例如：Qwen/Qwen3-0.6B → autorun_Qwen_Qwen3-0.6B
    _default_session_key = "autorun_" + MODEL_ID.replace("/", "_")
    SESSION_KEY   = os.getenv("SESSION_KEY", _default_session_key)

    print(f"\n模型:           {MODEL_ID}")
    print(f"设备:           {DEVICE}:{DEVICE_INDEX}")
    print(f"输出目录:        {OUTPUT_DIR}")
    print(f"HF cache:        {HF_CACHE}")
    print(f"HF endpoint:     {HF_ENDPOINT or '(default)'}")
    print(f"使用 session_key: {SESSION_KEY}")
    print(f"Agent timeout:   {TIMEOUT} 秒 ({TIMEOUT/60:.0f} 分钟)")
    print()

    test_task = build_task(MODEL_ID, DEVICE, DEVICE_INDEX, OUTPUT_DIR)
    result = run_agent_task(test_task, session_key=SESSION_KEY, timeout=TIMEOUT,
                            transcript_dest=f"{OUTPUT_DIR}/auto_run")

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
