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
AGENT_PASSWORD = "123"          # 建议改为 SSH 密钥认证

# open-claw 完整启动命令（在远端 docker 容器内执行）
def build_agent_cmd(model: str, task: str) -> str:
    return f"""
docker exec xpu-openclaw bash -c '
source /opt/openclaw-venv/bin/activate && \
export https_proxy=http://child-igk.intel.com:912 && \
export http_proxy=http://child-igk.intel.com:912 && \
export HF_HOME=/storage/lkk/cache && \
export AUTO_RUN_DEVICE=xpu && \
export AUTO_RUN_LOCAL=1 && \
export MINIMAX_API_KEY="sk-cp-xxx" && \
python batch_inference_test.py \
    --models {model} \
    --device xpu \
    --device-index 0 \
    --output-root /storage/lkk/inference
'"""

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "microsoft/phi-2",
]


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
    t.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
    return {"task_id": task_id}


async def run_agent(task_id: str, model: str, prompt: str, task: str):
    queue = task_queues[task_id]
    try:
        await queue.put(f"[SSH] Connecting to {AGENT_HOST} ...")

        async with asyncssh.connect(
            AGENT_HOST,
            username=AGENT_USER,
            password=AGENT_PASSWORD,
            known_hosts=None,          # 跳过 host key 检查（生产环境建议去掉）
        ) as conn:
            await queue.put(f"[SSH] Connected. Starting task: {task} | model: {model}")

            cmd = build_agent_cmd(model, task)

            async with conn.create_process(cmd) as proc:
                # 实时读取 stdout
                async for line in proc.stdout:
                    await queue.put(line.rstrip())
                # 同时捕获 stderr
                async for line in proc.stderr:
                    await queue.put(f"[stderr] {line.rstrip()}")

        await queue.put("[SSH] Task finished.")

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