# KernelAgent — open-claw Control Demo

A minimal full-stack demo: **SvelteKit** frontend + **FastAPI** backend with real-time SSE streaming.

## Quick Start

### 1. Start the backend

```bash
cd server
pip install fastapi uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### 2. Start the frontend

```bash
cd UI
npm install
npm run dev
```

### 3. Open browser

Visit **http://localhost:5173**

Select a model, enter a prompt, click **Auto-Test / Auto-Quant / Auto-Eval** to see real-time streaming output.

## Architecture

```
Browser (SvelteKit :5173)
  │
  ├─ POST /api/run ──────────► FastAPI (:8000)
  │                                │
  │                                └─ docker exec: open-claw agent
  │
  └─ GET  /api/stream/{id} ◄── SSE line-by-line stdout stream
```

## File Structure

```
KernelAgent/
├── server/
│   └── backend.py        # FastAPI main app (routes + SSE, docker exec runner)
├── UI/
│   ├── package.json
│   ├── svelte.config.js
│   ├── vite.config.ts
│   └── src/
│       ├── App.svelte     # Main page
│       └── main.ts
└── README.md
```

## Next Step

Update the docker container name and working directory constants at the top of `server/backend.py` to match your environment.
