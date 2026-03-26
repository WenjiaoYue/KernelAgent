# KernelAgent — open-claw Control Demo

A minimal full-stack demo: **SvelteKit** frontend + **FastAPI** backend with real-time SSE streaming.

## Quick Start

### 1. Start the backend

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Open browser

Visit **http://localhost:5173**

Select a model and task, enter a prompt, click **Run Demo** to see real-time streaming output.

Available tasks:
- `auto-test`, `auto-quant`, `auto-eval` for LLM agent flow
- `text-to-image` for image generation demo (`Tongyi-MAI/Z-Image-Turbo` recommended)

## Architecture

```
Browser (SvelteKit :5173)
  │
  ├─ POST /api/run ──────────► FastAPI (:8000)
  ├─ GET  /api/tasks ────────► task list for UI
  │                                │
  │                                └─ SSH + docker exec real agent command
  │
  └─ GET  /api/stream/{id} ◄── SSE line-by-line stdout stream
```

## File Structure

```
KernelAgent/
├── server/
│   ├── main.py           # FastAPI main app (routes + SSE)
│   ├── mock_agent.py     # Mock open-claw agent (line-by-line output, simulates slow task)
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── package.json
│   ├── svelte.config.js
│   ├── vite.config.js
│   └── src/
│       ├── app.html
│       └── routes/
│           └── +page.svelte   # Main page
└── README.md
```

## Next Step

Replace `server/mock_agent.py` invocation in `main.py` with your real open-claw agent command.
