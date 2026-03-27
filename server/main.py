"""
OpenClaw Pipeline Backend - FastAPI SSE Server

Provides:
  POST /api/run      → start a pipeline task, returns task_id
  GET  /api/stream/{task_id} → SSE stream of events
  GET  /api/models   → list of supported models
  GET  /api/stages   → list of pipeline stages

Runs pipeline scripts inside Docker container test_clawd via SSH.
Streams both process stdout/stderr and the openclaw agent session transcript.

Usage:
    pip install fastapi uvicorn asyncssh
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
import re
import shlex
import time
import uuid

import asyncssh
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

task_queues: dict[str, asyncio.Queue] = {}

# ── Agent 服务器 SSH 配置 ──────────────────────────────────────
AGENT_HOST = "h3c.sh.intel.com"
AGENT_USER = "kaokao"
AGENT_PASSWORD = "123"

AGENT_REPO_ROOT = "/kaokao/openclaw-triton-gen"
AGENT_WORKDIR   = f"{AGENT_REPO_ROOT}/examples/auto_run"
OUTPUT_ROOT     = "/storage/lkk/inference"
DOCKER_CONTAINER = "test_clawd"
DEFAULT_MINIMAX_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "sk-cp-LKAgYVTb1A-TrGbTtRar7C9HnFtfFxlNdVSTXk2rO9_aBzS5c1JMcZoRCxTmfnP2SHUGQUu3t7G38iIBaswe1p6eV_7hOOZy6KfZ2tVXSCbf2_bl1tZD4gs",
)

ALL_STAGES = ["auto_run", "auto_quant", "auto_eval"]

STAGE_SCRIPTS = {
    "auto_run":   "test_auto_run_skill.py",
    "auto_quant": "test_auto_quant_skill.py",
    "auto_eval":  "test_auto_eval_skill.py",
}

PIPELINE_SCRIPT = "test_auto_pipeline_skill.py"

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "microsoft/phi-2",
]

# Files expected per stage under <output_root>/<model_dir>/
STAGE_FILES = {
    "auto_run": [
        "auto_run/transcript.jsonl",
        "auto_run/inference_script.py",
        "auto_run/logs/inference.log",
        "auto_run/summary.md",
    ],
    "auto_quant": [
        "auto_quant/transcript.jsonl",
    ],
    "auto_eval": [
        "auto_eval/transcript.jsonl",
    ],
}


# ── Helpers ────────────────────────────────────────────────────

def model_to_dir(model: str) -> str:
    return model.replace("/", "_")


def base_env_exports() -> str:
    exports = [
        "export https_proxy=http://child-igk.intel.com:912",
        "export http_proxy=http://child-igk.intel.com:912",
        "export HF_HOME=/storage/lkk/cache",
    ]
    if DEFAULT_MINIMAX_KEY:
        exports.append(f"export MINIMAX_API_KEY={shlex.quote(DEFAULT_MINIMAX_KEY)}")
    return " && ".join(exports)


def build_pipeline_cmd(model: str, stages: list[str], device: str = "xpu", machine=None) -> str:
    """Build docker exec command for the given stages."""
    # Use machine config if provided, else fall back to module-level constants
    workdir     = machine.workdir     if machine else AGENT_WORKDIR
    output_root = machine.output_root if machine else OUTPUT_ROOT
    container   = machine.container   if machine else DOCKER_CONTAINER
    minimax_key = machine.minimax_key if machine else DEFAULT_MINIMAX_KEY

    quoted_model  = shlex.quote(model)
    quoted_output = shlex.quote(output_root)
    quoted_device = shlex.quote(device)

    selected = [s for s in ALL_STAGES if s in stages]
    if len(selected) == 1:
        script     = STAGE_SCRIPTS[selected[0]]
        stages_env = ""
    elif set(selected) == set(ALL_STAGES):
        script     = PIPELINE_SCRIPT
        stages_env = ""
    else:
        script     = PIPELINE_SCRIPT
        stages_env = f"export STAGES={shlex.quote(','.join(selected))} && "

    env_exports = [
        "export https_proxy=http://child-igk.intel.com:912",
        "export http_proxy=http://child-igk.intel.com:912",
        "export HF_HOME=/storage/lkk/cache",
    ]
    if minimax_key:
        env_exports.append(f"export MINIMAX_API_KEY={shlex.quote(minimax_key)}")
    env_str = " && ".join(env_exports)

    cmd = (
        f"cd {shlex.quote(workdir)} && "
        f"{env_str} && "
        f"export MODEL_ID={quoted_model} && "
        f"export DEVICE={quoted_device} && "
        f"export OUTPUT_ROOT={quoted_output} && "
        f"{stages_env}"
        f"python3 {script}"
    )
    return f"docker exec {container} bash -lc {shlex.quote(cmd)}"


async def enqueue_event(
    queue: asyncio.Queue,
    task_id: str,
    event_type: str,
    message: str,
    stage: str,
    level: str = "info",
    meta: dict | None = None,
):
    payload = {
        "task_id":      task_id,
        "timestamp_ms": int(time.time() * 1000),
        "type":         event_type,
        "stage":        stage,
        "level":        level,
        "message":      message,
    }
    if meta:
        payload["meta"] = meta
    await queue.put(payload)


# ── Process stdout/stderr streaming ───────────────────────────

_STAGE_RE = re.compile(
    r"\[(?:STAGE[:\s]+)?(auto_run|auto_quant|auto_eval)\]", re.IGNORECASE
)


async def stream_process_output(
    queue: asyncio.Queue, task_id: str, proc, stage: str
):
    """Stream stdout/stderr from the remote process."""
    current_stage = [stage]

    async def handle_stdout():
        async for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            m = _STAGE_RE.search(line)
            if m:
                current_stage[0] = m.group(1).lower()
                await enqueue_event(
                    queue, task_id, "status",
                    f"Stage started: {current_stage[0]}",
                    stage=current_stage[0],
                )
            level = "info"
            if "authentication_error" in line or "login fail" in line:
                level = "error"
            elif "error" in line.lower() and "traceback" not in line.lower():
                level = "warn"
            await enqueue_event(
                queue, task_id, "log", line,
                stage=current_stage[0], level=level,
            )

    async def handle_stderr():
        async for raw_line in proc.stderr:
            line = raw_line.rstrip()
            if not line:
                continue
            await enqueue_event(
                queue, task_id, "log", line,
                stage=current_stage[0], level="warn",
            )

    await asyncio.gather(handle_stdout(), handle_stderr())


# ── Transcript streaming ───────────────────────────────────────

def _session_key_for(model: str, stages: list[str]) -> str:
    """Compute the openclaw session key used by the stage script.
    Must match the logic in test_auto_*_skill.py.
    """
    slug = model.replace("/", "_")
    if len(stages) == 1:
        s = stages[0]
        if s == "auto_run":
            return f"autorun_{slug}"
        elif s == "auto_quant":
            return f"autoquant_{slug}_W4A16"
        elif s == "auto_eval":
            return f"autoeval_{slug}"
    # Pipeline script starts with auto_run session
    return f"pipeline_{slug}_autorun"


async def _resolve_session_file(conn, session_key: str, container: str = DOCKER_CONTAINER) -> str | None:
    """
    Resolve the live JSONL path for a session key inside the container.

    Tries both locations in ONE SSH call:
    1. {sessions_dir}/{session_key}.jsonl  — openclaw may store by session_key directly
    2. {sessions_dir}/{sessionId}.jsonl    — UUID looked up via sessions.json
    """
    py = (
        "import json,os,sys\n"
        "d='/root/.openclaw/agents/main/sessions'\n"
        f"k={json.dumps(session_key)}\n"
        "p=d+'/'+k+'.jsonl'\n"
        "if os.path.exists(p):\n"
        "    print(p);sys.exit(0)\n"
        "try:\n"
        "    data=json.load(open(d+'/sessions.json'))\n"
        "    for key in ['agent:main:'+k,k]:\n"
        "        sid=data.get(key,{}).get('sessionId','')\n"
        "        if sid:\n"
        "            pp=d+'/'+sid+'.jsonl'\n"
        "            if os.path.exists(pp):\n"
        "                print(pp);sys.exit(0)\n"
        "except Exception:\n"
        "    pass\n"
    )
    result = await conn.run(
        f"docker exec {container} python3 -c {shlex.quote(py)}",
        check=False,
    )
    path = result.stdout.strip()
    return path if path else None

def _extract_blocks(obj: dict) -> list[dict]:
    """
    Extract structured content blocks from one openclaw JSONL line.
    Returns a list of {"sub_type": ..., "role": ..., "text": ...} dicts.

    sub_type values:
      "text"        — agent's written response / summary
      "thinking"    — agent's internal reasoning (collapsible in UI)
      "tool_call"   — tool invocation: name + key arguments
      "tool_result" — tool output: stdout / file content

    openclaw session format:
      - type=message, role=user       → skip (initial prompt, not agent activity)
      - type=message, role=assistant  → thinking / text / toolCall content blocks
      - type=message, role=toolResult → text content is the tool output
    """
    msg  = obj.get("message", obj)
    role = msg.get("role", obj.get("type", "agent"))

    # Skip user messages — they are just the initial prompt, not agent activity
    if role == "user":
        return []

    content = msg.get("content", "")

    if not isinstance(content, list):
        text = str(content).strip()
        return [{"sub_type": "text", "role": role, "text": text}] if text else []

    blocks = []
    is_tool_result_msg = (role == "toolResult")

    for c in content:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type", "")

        if ctype == "text":
            text = c.get("text", "").strip()
            if text:
                # In role=toolResult messages, the text IS the tool output
                if is_tool_result_msg:
                    if len(text) > 2000:
                        text = "...(truncated)...\n" + text[-1800:]
                    blocks.append({"sub_type": "tool_result", "role": role, "text": text})
                else:
                    blocks.append({"sub_type": "text", "role": role, "text": text})

        elif ctype == "thinking":
            thinking = c.get("thinking", "").strip()
            if thinking:
                blocks.append({"sub_type": "thinking", "role": role, "text": thinking})

        elif ctype == "toolCall":
            name = c.get("name", "?")
            args = c.get("arguments", {})
            # For exec, show the command directly; for write, show path; otherwise dump args
            if name == "exec":
                cmd = args.get("command", "")
                display = f"$ {cmd[:400]}"
            elif name == "write":
                path = args.get("path", args.get("file", "?"))
                display = f"write → {path}"
            elif name == "read":
                path = args.get("path", args.get("file", "?"))
                display = f"read ← {path}"
            else:
                display = json.dumps(args, ensure_ascii=False)[:300]
            blocks.append({"sub_type": "tool_call", "role": role,
                           "text": display, "tool_name": name})

        elif ctype == "toolResult":
            inner = c.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", "") for x in inner if isinstance(x, dict)
                )
            text = str(inner).strip()
            if text:
                # Truncate very long results but keep tail (most recent output is at end)
                if len(text) > 2000:
                    text = "...(truncated)...\n" + text[-1800:]
                blocks.append({"sub_type": "tool_result", "role": role, "text": text})

        elif "text" in c:
            text = str(c["text"]).strip()
            if text:
                blocks.append({"sub_type": "text", "role": role, "text": text})

    return blocks


async def stream_transcript(
    queue: asyncio.Queue,
    task_id: str,
    conn,
    stage: str,
    stop_event: asyncio.Event,
    model: str,
    stages: list[str],
    workspace_dir: str,
    container: str = DOCKER_CONTAINER,
    job_start_ms: int = 0,
):
    """
    Tail the REAL-TIME openclaw session JSONL inside Docker.

    Strategy:
    - Wait for the session file to appear (openclaw creates it on first run).
    - Snapshot its current byte size immediately after finding it.
    - Tail from byte_offset = size + 1, so only bytes written by THIS run are shown.
    - openclaw appends to the session file across runs (key-based sessions are not reset),
      so the byte-offset approach reliably skips all previous content.
    """
    session_key = _session_key_for(model, stages)

    await enqueue_event(
        queue, task_id, "status",
        f"Waiting for session '{session_key}'…",
        stage=stage,
    )

    # Wait up to 90s for the session file to be created by openclaw.
    # One SSH call checks both {session_key}.jsonl and UUID-based path.
    session_file = None
    for i in range(180):  # 180 × 0.5s = 90s
        if stop_event.is_set():
            return
        session_file = await _resolve_session_file(conn, session_key, container)
        if session_file:
            break
        # Emit a periodic status so Pipeline Timeline shows progress (every 10s)
        if i > 0 and i % 20 == 0:
            elapsed = i // 2
            await enqueue_event(
                queue, task_id, "status",
                f"Still waiting for session '{session_key}'… ({elapsed}s)",
                stage=stage,
            )
        await asyncio.sleep(0.5)

    if not session_file:
        # Fallback: try the workspace output transcript.jsonl files directly.
        # The pipeline may write the transcript to the output dir rather than
        # the openclaw sessions dir (or under a different session key).
        for trial in [
            f"{workspace_dir}/transcript.jsonl",
            f"{workspace_dir}/auto_run/transcript.jsonl",
        ]:
            r = await conn.run(
                f"docker exec {container} bash -c "
                f"'test -f {shlex.quote(trial)} && echo yes || echo no'",
                check=False,
            )
            if r.stdout.strip() == "yes":
                session_file = trial
                await enqueue_event(
                    queue, task_id, "status",
                    f"Session file not found; falling back to {trial}",
                    stage=stage, level="warn",
                )
                break

    if not session_file:
        await enqueue_event(
            queue, task_id, "status",
            f"Session '{session_key}' not found; no transcript available.",
            stage=stage, level="warn",
        )
        return

    # Snapshot current file size so we only tail NEW bytes written by this run.
    stat_result = await conn.run(
        f"docker exec {container} stat -c%s {shlex.quote(session_file)} 2>/dev/null || echo 0",
        check=False,
    )
    try:
        byte_offset = int(stat_result.stdout.strip()) + 1
    except ValueError:
        byte_offset = 1

    await enqueue_event(
        queue, task_id, "status",
        f"Tailing session '{session_key}' from byte {byte_offset}",
        stage=stage,
    )

    tail_cmd = (
        f"docker exec {container} "
        f"tail -c +{byte_offset} --follow=name --retry --sleep-interval=0.2 {shlex.quote(session_file)}"
    )
    last_read_path: str | None = None  # track path of most recent 'read' tool_call
    try:
        async with conn.create_process(tail_cmd) as proc:
            async for raw_line in proc.stdout:
                # Do NOT check stop_event here — let cancel() drain the buffer.
                # Breaking early here would discard all buffered lines.
                line = raw_line.rstrip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    blocks = _extract_blocks(obj)
                    for block in blocks:
                        meta = {
                            "sub_type":  block["sub_type"],
                            "role":      block["role"],
                            "tool_name": block.get("tool_name"),
                        }
                        if block["sub_type"] == "tool_call" and block.get("tool_name") == "read":
                            # Extract path from "read ← /path/to/file"
                            txt = block["text"]
                            last_read_path = txt.split(" ← ", 1)[1] if " ← " in txt else None
                        elif block["sub_type"] == "tool_result":
                            if last_read_path:
                                meta["read_path"] = last_read_path
                            last_read_path = None  # consume after first tool_result
                        await enqueue_event(
                            queue, task_id, "transcript",
                            block["text"],
                            stage=stage,
                            meta=meta,
                        )
                        if block["sub_type"] == "tool_call":
                            tool = block.get("tool_name", "tool")
                            await enqueue_event(
                                queue, task_id, "log",
                                f"[agent:{tool}] {block['text']}",
                                stage=stage, level="info",
                            )
                except json.JSONDecodeError:
                    pass
    except (asyncssh.Error, asyncio.CancelledError):
        pass


# ── Workspace file polling ─────────────────────────────────────

async def poll_workspace_files(
    queue: asyncio.Queue,
    task_id: str,
    conn,
    workspace_dir: str,
    stages: list[str],
    stop_event: asyncio.Event,
    container: str = DOCKER_CONTAINER,
):
    """Periodically check which output files exist and emit file_created events.

    Snapshots file existence BEFORE the run starts so we only report NEW files,
    not files left over from a previous run.
    """
    watched: dict[str, bool] = {}
    for stage in stages:
        for rel in STAGE_FILES.get(stage, []):
            path = f"{workspace_dir}/{rel}"
            result = await conn.run(
                f"docker exec {container} bash -c "
                f"'test -f {shlex.quote(path)} && echo yes || echo no'",
                check=False,
            )
            # Mark pre-existing files as already seen so they are NOT reported
            watched[rel] = result.stdout.strip() == "yes"

    while not stop_event.is_set():
        await asyncio.sleep(1)
        for rel, seen in list(watched.items()):
            if seen:
                continue
            path = f"{workspace_dir}/{rel}"
            # FIX: run test -f inside the docker container, not on the SSH host
            result = await conn.run(
                f"docker exec {container} bash -c "
                f"'test -f {shlex.quote(path)} && echo yes || echo no'",
                check=False,
            )
            if result.stdout.strip() == "yes":
                watched[rel] = True
                file_stage = rel.split("/")[0]
                await enqueue_event(
                    queue, task_id, "file_created", f"File ready: {rel}",
                    stage=file_stage,
                    meta={"path": rel, "full_path": path},
                )


# ── API endpoints ──────────────────────────────────────────────

ALL_DEVICES = ["xpu", "cuda"]


class MachineConfig(BaseModel):
    host:        str = AGENT_HOST
    user:        str = AGENT_USER
    password:    str = AGENT_PASSWORD
    repo_root:   str = AGENT_REPO_ROOT
    workdir:     str = AGENT_WORKDIR
    output_root: str = OUTPUT_ROOT
    container:   str = DOCKER_CONTAINER
    minimax_key: str = DEFAULT_MINIMAX_KEY


class RunRequest(BaseModel):
    model: str
    stages: list[str] = ["auto_run", "auto_quant", "auto_eval"]
    device: str = "xpu"
    machine: MachineConfig = MachineConfig()


@app.get("/api/models")
async def get_models():
    return {"models": MODELS}


@app.get("/api/stages")
async def get_stages():
    return {"stages": ALL_STAGES}


class DebugRequest(BaseModel):
    host:        str = AGENT_HOST
    user:        str = AGENT_USER
    password:    str = AGENT_PASSWORD
    container:   str = DOCKER_CONTAINER
    output_root: str = OUTPUT_ROOT
    model:       str = "Qwen/Qwen3-0.6B"
    stages:      list[str] = ["auto_run"]


@app.post("/api/debug")
async def debug_check(req: DebugRequest):
    """
    Diagnostic endpoint: SSH into the machine, check session files and
    transcript file existence so the user can see exactly what paths are
    being used and whether the files exist.
    """
    out: dict = {
        "session_key": _session_key_for(req.model, req.stages),
        "model_dir":   model_to_dir(req.model),
        "ssh":         None,
        "session_files": [],
        "sessions_json_keys": [],
        "resolved_session_file": None,
        "transcript_paths": {},
    }
    try:
        async with asyncssh.connect(
            req.host, username=req.user, password=req.password, known_hosts=None
        ) as conn:
            out["ssh"] = "ok"

            sessions_dir = "/root/.openclaw/agents/main/sessions"

            # List all session JSONL files inside Docker
            r = await conn.run(
                f"docker exec {req.container} bash -c "
                f"'ls {sessions_dir}/ 2>/dev/null | grep -E \\.jsonl$ | head -40'",
                check=False,
            )
            out["session_files"] = [line for line in r.stdout.strip().split("\n") if line]

            # Read sessions.json keys to see how sessions are named
            py_keys = (
                "import json,os\n"
                f"p='{sessions_dir}/sessions.json'\n"
                "d=json.load(open(p)) if os.path.exists(p) else {{}}\n"
                "print('\\n'.join(list(d.keys())[:40]))\n"
            )
            r2 = await conn.run(
                f"docker exec {req.container} python3 -c {shlex.quote(py_keys)}",
                check=False,
            )
            out["sessions_json_keys"] = [line for line in r2.stdout.strip().split("\n") if line]

            # Try to resolve the session file using the same logic as stream_transcript
            sf = await _resolve_session_file(conn, out["session_key"], req.container)
            out["resolved_session_file"] = sf
            if sf:
                r3 = await conn.run(
                    f"docker exec {req.container} wc -l {shlex.quote(sf)} 2>/dev/null || echo 0",
                    check=False,
                )
                out["session_file_lines"] = r3.stdout.strip()

            # Check workspace transcript files
            model_dir = model_to_dir(req.model)
            for trial in [
                f"{req.output_root}/{model_dir}/transcript.jsonl",
                f"{req.output_root}/{model_dir}/auto_run/transcript.jsonl",
                f"{req.output_root}/{model_dir}/auto_quant/transcript.jsonl",
                f"{req.output_root}/{model_dir}/auto_eval/transcript.jsonl",
            ]:
                r4 = await conn.run(
                    f"docker exec {req.container} bash -c "
                    f"'test -f {shlex.quote(trial)} && wc -l {shlex.quote(trial)} || echo NOT_FOUND'",
                    check=False,
                )
                out["transcript_paths"][trial] = r4.stdout.strip()

    except Exception as e:
        out["ssh"] = f"FAILED: {e}"

    return out


@app.post("/api/run")
async def run_task(req: RunRequest):
    invalid = [s for s in req.stages if s not in ALL_STAGES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown stages: {invalid}")
    if not req.stages:
        raise HTTPException(status_code=400, detail="At least one stage is required")

    task_id = str(uuid.uuid4())
    task_queues[task_id] = asyncio.Queue()
    t = asyncio.create_task(run_pipeline(task_id, req.model, req.stages, req.device, req.machine))
    t.add_done_callback(
        lambda t: t.exception() if not t.cancelled() and t.exception() else None
    )
    return {"task_id": task_id}


async def run_pipeline(task_id: str, model: str, stages: list[str], device: str = "xpu", machine: MachineConfig | None = None):
    if machine is None:
        machine = MachineConfig()
    queue = task_queues[task_id]
    workspace_dir = f"{machine.output_root}/{model_to_dir(model)}"
    initial_stage = stages[0] if len(stages) == 1 else "pipeline"

    try:
        await enqueue_event(
            queue, task_id, "status",
            f"Connecting to {machine.host}", stage="connect",
        )

        async with asyncssh.connect(
            machine.host,
            username=machine.user,
            password=machine.password,
            known_hosts=None,
        ) as conn:
            await enqueue_event(
                queue, task_id, "status", "SSH connected", stage="connect",
            )

            run_cmd     = build_pipeline_cmd(model, stages, device, machine)
            stage_label = "+".join(stages)

            await enqueue_event(
                queue, task_id, "status",
                f"Starting pipeline stages: {stage_label}",
                stage=initial_stage,
                meta={"model": model, "stages": stages, "workspace_dir": workspace_dir},
            )

            if not machine.minimax_key:
                await enqueue_event(
                    queue, task_id, "status",
                    "MINIMAX_API_KEY is empty – may cause 401 errors.",
                    stage=initial_stage, level="warn",
                )

            stop_poll       = asyncio.Event()
            stop_transcript = asyncio.Event()

            poll_task = asyncio.create_task(
                poll_workspace_files(
                    queue, task_id, conn, workspace_dir, stages, stop_poll,
                    container=machine.container,
                )
            )
            # Start transcript tailing BEFORE launching the process so we
            # can detect the moment the new session file appears.
            transcript_task = asyncio.create_task(
                stream_transcript(
                    queue, task_id, conn, initial_stage, stop_transcript,
                    model=model, stages=stages,
                    workspace_dir=workspace_dir,
                    container=machine.container,
                    job_start_ms=int(time.time() * 1000),
                )
            )

            async with conn.create_process(run_cmd) as proc:
                await stream_process_output(
                    queue, task_id, proc, stage=initial_stage,
                )
                result = await proc.wait()

            stop_poll.set()
            # Give the transcript task time to drain remaining buffered lines
            # before cancelling. tail --sleep-interval=0.2 means all written
            # bytes arrive within <1s; 8s is a safe margin for large transcripts.
            await asyncio.sleep(8)
            stop_transcript.set()
            transcript_task.cancel()
            try:
                await transcript_task
            except asyncio.CancelledError:
                pass
            await poll_task

            if result.exit_status == 0:
                await enqueue_event(
                    queue, task_id, "done",
                    "Pipeline finished successfully",
                    stage=initial_stage,
                    meta={
                        "exit_status":  result.exit_status,
                        "stages":       stages,
                        "workspace_dir": workspace_dir,
                    },
                )
            else:
                await enqueue_event(
                    queue, task_id, "error",
                    f"Pipeline failed with exit code {result.exit_status}",
                    stage=initial_stage, level="error",
                    meta={"exit_status": result.exit_status},
                )

    except asyncssh.Error as e:
        logger.exception("SSH error for task %s", task_id)
        await enqueue_event(
            queue, task_id, "error",
            f"SSH failed: {e}", stage="connect", level="error",
        )
    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await enqueue_event(
            queue, task_id, "error",
            str(e), stage="runtime", level="error",
        )
    finally:
        await queue.put(None)


# ── SSE endpoint ───────────────────────────────────────────────

@app.get("/api/stream/{task_id}")
async def stream(task_id: str):
    queue = task_queues.get(task_id)
    if not queue:
        raise HTTPException(status_code=404, detail="task not found")

    async def generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # SSE comment keeps the connection alive through proxies/browsers
                    yield ": heartbeat\n\n"
                    continue
                if item is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            task_queues.pop(task_id, None)

    return StreamingResponse(generator(), media_type="text/event-stream")
