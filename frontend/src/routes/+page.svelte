<script>
  import { onMount } from 'svelte';

  let models = [];
  let selectedModel = '';
  let prompt = '';
  let logs = [];
  let running = false;
  let currentTask = '';
  let done = false;
  let error = '';

  onMount(async () => {
    const res = await fetch('http://localhost:8000/api/models');
    const data = await res.json();
    models = data.models;
    selectedModel = models[0];
  });

  async function runTask(task) {
    logs = [];
    done = false;
    error = '';
    running = true;
    currentTask = task;

    const res = await fetch('http://localhost:8000/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel, prompt, task }),
    });
    const { task_id } = await res.json();

    const evtSource = new EventSource(`http://localhost:8000/api/stream/${task_id}`);
    evtSource.onmessage = (e) => {
      if (e.data === '[DONE]') {
        evtSource.close();
        running = false;
        done = true;
        return;
      }
      logs = [...logs, e.data];
    };
    evtSource.onerror = () => {
      evtSource.close();
      running = false;
      error = 'Stream connection failed. Please try again.';
    };
  }
</script>

<main>
  <h1>🤖 open-claw Agent Control</h1>

  <div class="controls">
    <label>
      Model:
      <select bind:value={selectedModel}>
        {#each models as m}
          <option value={m}>{m}</option>
        {/each}
      </select>
    </label>

    <label>
      Prompt:
      <input type="text" bind:value={prompt} placeholder="Enter your prompt..." />
    </label>

    <div class="buttons">
      <button disabled={running} on:click={() => runTask('auto-test')}>
        {running && currentTask === 'auto-test' ? '⏳ Running...' : '▶ Auto-Test'}
      </button>
      <button disabled={running} on:click={() => runTask('auto-quant')}>
        {running && currentTask === 'auto-quant' ? '⏳ Running...' : '▶ Auto-Quant'}
      </button>
      <button disabled={running} on:click={() => runTask('auto-eval')}>
        {running && currentTask === 'auto-eval' ? '⏳ Running...' : '▶ Auto-Eval'}
      </button>
    </div>
  </div>

  <div class="log-panel">
    <div class="log-header">
      📋 Live Output
      {#if done}<span class="done">✅ Task completed</span>{/if}
    </div>
    {#if error}
      <div class="error-banner">{error}</div>
    {/if}
    <div class="log-body">
      {#each logs as line}
        <div class="log-line">{line}</div>
      {/each}
    </div>
  </div>
</main>

<style>
  main {
    max-width: 800px;
    margin: 40px auto;
    font-family: sans-serif;
    padding: 0 20px;
  }

  h1 { margin-bottom: 24px; }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 24px;
  }

  label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 500;
  }

  select, input {
    padding: 6px 10px;
    border: 1px solid #ccc;
    border-radius: 6px;
    flex: 1;
    font-size: 14px;
  }

  .buttons {
    display: flex;
    gap: 10px;
  }

  button {
    padding: 8px 18px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
  }

  button:disabled {
    background: #93c5fd;
    cursor: not-allowed;
  }

  button:hover:not(:disabled) {
    background: #2563eb;
  }

  .log-panel {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
  }

  .log-header {
    background: #f9fafb;
    padding: 10px 16px;
    font-weight: 600;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .done { color: #16a34a; font-weight: 600; }

  .error-banner {
    background: #fef2f2;
    color: #dc2626;
    padding: 8px 16px;
    border-bottom: 1px solid #fecaca;
    font-size: 13px;
  }

  .log-body {
    height: 320px;
    overflow-y: auto;
    padding: 12px 16px;
    background: #111827;
  }

  .log-line {
    font-family: monospace;
    font-size: 13px;
    color: #d1fae5;
    line-height: 1.6;
    white-space: pre-wrap;
  }
</style>
