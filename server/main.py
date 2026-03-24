import asyncio
import logging
import sys
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

task_queues: dict[str, asyncio.Queue] = {}

MODELS = ["gpt2", "llama-7b", "mistral-7b", "phi-2"]

AGENT_SCRIPT = Path(__file__).parent / "mock_agent.py"


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
    task = asyncio.create_task(run_agent(task_id, req.model, req.prompt, req.task))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
    return {"task_id": task_id}


async def run_agent(task_id: str, model: str, prompt: str, task: str):
    queue = task_queues[task_id]
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(AGENT_SCRIPT),
            "--model", model,
            "--prompt", prompt,
            "--task", task,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            await queue.put(line.decode().rstrip())
        await proc.wait()
    except Exception:
        logger.exception("Agent task %s failed", task_id)
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
