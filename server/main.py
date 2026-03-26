import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncssh

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

AGENT_REPO_ROOT = "/wenjiao/openclaw-triton-gen"
AGENT_WORKDIR = f"{AGENT_REPO_ROOT}/examples/auto_run"
OUTPUT_ROOT = "/storage/lkk/inference"
DOCKER_CONTAINER = "xpu-openclaw"
DEFAULT_MINIMAX_KEY = os.getenv("MINIMAX_API_KEY", "")

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "microsoft/phi-2",
]

TASKS = [
    {"id": "auto-test", "label": "Auto-Test", "kind": "llm"},
    {"id": "auto-quant", "label": "Auto-Quant", "kind": "llm"},
    {"id": "auto-eval", "label": "Auto-Eval", "kind": "llm"},
    {"id": "text-to-image", "label": "Text-to-Image", "kind": "image"},
]

def model_to_dir(model: str) -> str:
    """Qwen/Qwen3-0.6B -> Qwen_Qwen3-0.6B"""
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


def build_agent_cmd(model: str) -> str:
    quoted_model = shlex.quote(model)
    quoted_output = shlex.quote(OUTPUT_ROOT)
    cmd = (
        f"cd {shlex.quote(AGENT_WORKDIR)} && "
        "source /opt/openclaw-venv/bin/activate && "
        f"{base_env_exports()} && "
        "export AUTO_RUN_DEVICE=xpu && "
        "export AUTO_RUN_LOCAL=1 && "
        "python batch_inference_test.py"
        f" --models {quoted_model}"
        " --device xpu"
        " --device-index 0"
        f" --output-root {quoted_output}"
    )
    return f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(cmd)}"


def build_text_to_image_cmd(model: str, prompt: str) -> tuple[str, str]:
    safe_model = shlex.quote(model)
    safe_prompt = shlex.quote(prompt or "a cup of coffee on the table")
    output_file = f"{OUTPUT_ROOT}/z_image_turbo_output_{uuid.uuid4().hex[:8]}.png"
    safe_output = shlex.quote(output_file)
    cmd = (
        f"cd {shlex.quote(AGENT_REPO_ROOT)} && "
        "source /opt/openclaw-venv/bin/activate && "
        f"{base_env_exports()} && "
        "python examples/offline_inference/text_to_image/text_to_image.py"
        f" --model {safe_model}"
        f" --prompt {safe_prompt}"
        f" --output {safe_output}"
        " --seed 42"
        " --cfg-scale 4.0"
        " --guidance-scale 1.0"
        " --num-inference-steps 50"
    )
    return f"docker exec {DOCKER_CONTAINER} bash -lc {shlex.quote(cmd)}", output_file


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
        "task_id": task_id,
        "timestamp_ms": int(time.time() * 1000),
        "type": event_type,
        "stage": stage,
        "level": level,
        "message": message,
    }
    if meta:
        payload["meta"] = meta
    await queue.put(payload)


async def stream_process_output(queue: asyncio.Queue, task_id: str, proc, stage: str):
    async def handle_stdout():
        async for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            level = "info"
            if "authentication_error" in line or "login fail" in line:
                level = "error"
            await enqueue_event(queue, task_id, "log", line, stage=stage, level=level)

    async def handle_stderr():
        async for raw_line in proc.stderr:
            line = raw_line.rstrip()
            if not line:
                continue
            await enqueue_event(queue, task_id, "log", line, stage=stage, level="warn")

    await asyncio.gather(handle_stdout(), handle_stderr())

class RunRequest(BaseModel):
    model: str
    prompt: str = ""
    task: str = "auto-test"

@app.get("/api/models")
async def get_models():
    return {"models": MODELS}


@app.get("/api/tasks")
async def get_tasks():
    return {"tasks": TASKS}

@app.post("/api/run")
async def run_task(req: RunRequest):
    task_id = str(uuid.uuid4())
    task_queues[task_id] = asyncio.Queue()
    t = asyncio.create_task(run_agent(task_id, req.model, req.prompt, req.task))
    t.add_done_callback(
        lambda t: t.exception() if not t.cancelled() and t.exception() else None
    )
    return {"task_id": task_id}

async def run_agent(task_id: str, model: str, prompt: str, task: str):
    queue = task_queues[task_id]
    try:
        await enqueue_event(queue, task_id, "status", f"Connecting to {AGENT_HOST}", stage="connect")

        async with asyncssh.connect(
            AGENT_HOST,
            username=AGENT_USER,
            password=AGENT_PASSWORD,
            known_hosts=None,
        ) as conn:
            await enqueue_event(queue, task_id, "status", "SSH connected", stage="connect")

            if task == "text-to-image":
                run_cmd, output_file = build_text_to_image_cmd(model, prompt)
                stage = "text-to-image"
                await enqueue_event(
                    queue,
                    task_id,
                    "status",
                    f"Running text-to-image model: {model}",
                    stage=stage,
                    meta={"task": task, "model": model, "output": output_file},
                )
            else:
                run_cmd = build_agent_cmd(model)
                output_dir = f"{OUTPUT_ROOT}/{model_to_dir(model)}"
                stage = "agent"
                await enqueue_event(
                    queue,
                    task_id,
                    "status",
                    f"Running agent task: {task}",
                    stage=stage,
                    meta={"task": task, "model": model, "output_dir": output_dir},
                )

            if not DEFAULT_MINIMAX_KEY and task != "text-to-image":
                await enqueue_event(
                    queue,
                    task_id,
                    "status",
                    "MINIMAX_API_KEY is empty. This usually causes 401 authentication errors.",
                    stage=stage,
                    level="warn",
                )

            async with conn.create_process(run_cmd) as proc:
                await stream_process_output(queue, task_id, proc, stage=stage)
                result = await proc.wait()

            if result.exit_status == 0:
                await enqueue_event(
                    queue,
                    task_id,
                    "done",
                    "Task finished successfully",
                    stage=stage,
                    meta={"exit_status": result.exit_status},
                )
            else:
                await enqueue_event(
                    queue,
                    task_id,
                    "error",
                    f"Task failed with exit code {result.exit_status}",
                    stage=stage,
                    level="error",
                    meta={"exit_status": result.exit_status},
                )

    except asyncssh.Error as e:
        logger.exception("SSH error for task %s", task_id)
        await enqueue_event(queue, task_id, "error", f"SSH failed: {e}", stage="connect", level="error")
    except Exception as e:
        logger.exception("Agent task %s failed", task_id)
        await enqueue_event(queue, task_id, "error", str(e), stage="runtime", level="error")
    finally:
        await queue.put(None)

@app.get("/api/stream/{task_id}")
async def stream(task_id: str):
    queue = task_queues.get(task_id)
    if not queue:
        raise HTTPException(status_code=404, detail="task not found")

    async def generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            task_queues.pop(task_id, None)

    return StreamingResponse(generator(), media_type="text/event-stream")