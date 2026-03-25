import asyncio
import logging
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

AGENT_WORKDIR = "/wenjiao/openclaw-triton-gen/examples/auto_run"
OUTPUT_ROOT = "/storage/lkk/inference"
DOCKER_CONTAINER = "xpu-openclaw"

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "microsoft/phi-2",
]

def model_to_dir(model: str) -> str:
    """Qwen/Qwen3-0.6B -> Qwen_Qwen3-0.6B"""
    return model.replace("/", "_")

def build_run_cmd(model: str) -> str:
    """后台启动 batch_inference_test.py，nohup 让它在 SSH 断开后继续跑"""
    return (
        f"docker exec -d {DOCKER_CONTAINER} bash -c '"
        f"cd {AGENT_WORKDIR} && "
        f"source /opt/openclaw-venv/bin/activate && "
        f"export https_proxy=http://child-igk.intel.com:912 && "
        f"export http_proxy=http://child-igk.intel.com:912 && "
        f"export HF_HOME=/storage/lkk/cache && "
        f"export AUTO_RUN_DEVICE=xpu && "
        f"export AUTO_RUN_LOCAL=1 && "
        f"export MINIMAX_API_KEY=\"sk-cp-xxx\" && "
        f"python batch_inference_test.py"
        f" --models {model}"
        f" --device xpu"
        f" --device-index 0"
        f" --output-root {OUTPUT_ROOT}"
        f"'"
    )

def build_tail_cmd(model: str) -> str:
    """等 session.jsonl 出现后实时 tail"""
    session_file = f"{OUTPUT_ROOT}/{model_to_dir(model)}/session.jsonl"
    return (
        f"docker exec {DOCKER_CONTAINER} bash -c '"
        f"until [ -f {session_file} ]; do sleep 1; done && "
        f"tail -f {session_file}"
        f"'"
    )

class RunRequest(BaseModel):
    model: str
    prompt: str = ""
    task: str = "auto-test"

@app.get("/api/models")
async def get_models():
    return {"models": MODELS}

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
        await queue.put(f"[SSH] Connecting to {AGENT_HOST} ...")

        async with asyncssh.connect(
            AGENT_HOST,
            username=AGENT_USER,
            password=AGENT_PASSWORD,
            known_hosts=None,
        ) as conn:
            await queue.put(f"[SSH] Connected. Launching agent for model: {model}")

            # ① 后台启动 batch_inference_test.py（-d 模式，立即返回）
            run_cmd = build_run_cmd(model)
            await conn.run(run_cmd)
            await queue.put(f"[Agent] Task launched in background. Waiting for session.jsonl ...")

            # ② tail -f session.jsonl，实时推送新增行
            tail_cmd = build_tail_cmd(model)
            async with conn.create_process(tail_cmd) as proc:
                async for line in proc.stdout:
                    await queue.put(line.rstrip())
                async for line in proc.stderr:
                    await queue.put(f"[stderr] {line.rstrip()}")

    except asyncssh.Error as e:
        logger.exception("SSH error for task %s", task_id)
        await queue.put(f"[ERROR] SSH failed: {e}")
    except Exception as e:
        logger.exception("Agent task %s failed", task_id)
        await queue.put(f"[ERROR] {e}")
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
                line = await queue.get()
                if line is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {line}\n\n"
        finally:
            task_queues.pop(task_id, None)

    return StreamingResponse(generator(), media_type="text/event-stream")