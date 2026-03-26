<script>
  import { onMount } from 'svelte';

  let models = [];
  let tasks = [];
  let selectedModel = '';
  let selectedTask = 'auto-test';
  let prompt = '';
  let timeline = [];
  let logs = [];
  let errors = [];
  let summary = null;
  let running = false;
  let done = false;
  let connectionError = '';
  let stageExpanded = {};
  let runHistory = [];
  let currentRunStartedAt = 0;
  let hadErrorThisRun = false;

  $: totalRuns = runHistory.length;
  $: successRuns = runHistory.filter((item) => item.success).length;
  $: successRate = totalRuns === 0 ? 0 : Math.round((successRuns / totalRuns) * 100);
  $: avgDurationMs = totalRuns === 0
    ? 0
    : Math.round(runHistory.reduce((sum, item) => sum + item.durationMs, 0) / totalRuns);

  $: stageMetrics = buildStageMetrics([...timeline, ...logs, ...errors]);

  onMount(async () => {
    const [modelRes, taskRes] = await Promise.all([
      fetch('http://localhost:8000/api/models'),
      fetch('http://localhost:8000/api/tasks'),
    ]);
    const modelData = await modelRes.json();
    const taskData = await taskRes.json();
    models = modelData.models;
    tasks = taskData.tasks;
    selectedModel = models[0];
    selectedTask = tasks[0]?.id ?? 'auto-test';
  });

  function parseEvent(raw) {
    try {
      const event = JSON.parse(raw);
      if (!event.timestamp_ms) {
        event.timestamp_ms = Date.now();
      }
      return event;
    } catch {
      return {
        type: 'log',
        stage: 'legacy',
        level: 'info',
        message: raw,
        timestamp_ms: Date.now(),
      };
    }
  }

  function toClock(timestampMs) {
    return new Date(timestampMs).toLocaleTimeString();
  }

  function formatDuration(durationMs) {
    if (!durationMs || durationMs < 1000) {
      return `${durationMs || 0}ms`;
    }
    return `${(durationMs / 1000).toFixed(1)}s`;
  }

  function buildStageMetrics(events) {
    const stageMap = new Map();
    for (const event of events) {
      const stage = event.stage || 'unknown';
      const ts = event.timestamp_ms || Date.now();
      if (!stageMap.has(stage)) {
        stageMap.set(stage, {
          stage,
          start: ts,
          end: ts,
          events: [],
          warns: 0,
          errors: 0,
        });
      }
      const bucket = stageMap.get(stage);
      bucket.start = Math.min(bucket.start, ts);
      bucket.end = Math.max(bucket.end, ts);
      if (event.level === 'warn') bucket.warns += 1;
      if (event.level === 'error' || event.type === 'error') bucket.errors += 1;
      bucket.events.push(event);
    }
    return Array.from(stageMap.values()).sort((a, b) => a.start - b.start);
  }

  function toggleStage(stage) {
    stageExpanded = {
      ...stageExpanded,
      [stage]: !(stageExpanded[stage] ?? true),
    };
  }

  async function runTask() {
    if (!selectedModel || !selectedTask) return;

    timeline = [];
    logs = [];
    errors = [];
    summary = null;
    done = false;
    connectionError = '';
    hadErrorThisRun = false;
    currentRunStartedAt = Date.now();
    running = true;

    const res = await fetch('http://localhost:8000/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel, prompt, task: selectedTask }),
    });
    const { task_id } = await res.json();

    const evtSource = new EventSource(`http://localhost:8000/api/stream/${task_id}`);
    evtSource.onmessage = (e) => {
      if (e.data === '[DONE]') {
        evtSource.close();
        running = false;
        done = true;
        const durationMs = Date.now() - currentRunStartedAt;
        runHistory = [
          ...runHistory,
          {
            success: !hadErrorThisRun,
            durationMs,
            task: selectedTask,
            model: selectedModel,
          },
        ];
        return;
      }

      const event = parseEvent(e.data);
      if (event.type === 'status') {
        timeline = [...timeline, event];
      } else if (event.type === 'error') {
        hadErrorThisRun = true;
        errors = [...errors, event];
      } else if (event.type === 'done') {
        summary = event;
      } else {
        logs = [...logs, event];
      }

      if (event.level === 'error' && event.type !== 'error') {
        hadErrorThisRun = true;
        errors = [...errors, event];
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      running = false;
      hadErrorThisRun = true;
      connectionError = 'Stream connection failed. Please try again.';
    };
  }

  function levelClass(level) {
    if (level === 'error') return 'level-error';
    if (level === 'warn') return 'level-warn';
    return 'level-info';
  }
</script>

<main>
  <h1>🤖 open-claw Agent Demo Console</h1>

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
      Task:
      <select bind:value={selectedTask}>
        {#each tasks as task}
          <option value={task.id}>{task.label}</option>
        {/each}
      </select>
    </label>

    <label>
      Prompt:
      <input type="text" bind:value={prompt} placeholder="Enter your prompt..." />
    </label>

    <div class="buttons">
      <button disabled={running} on:click={runTask}>
        {running ? '⏳ Running...' : '▶ Run Demo'}
      </button>
    </div>
  </div>

  <section class="panel">
    <div class="panel-header">📊 Demo KPIs</div>
    <div class="kpi-grid">
      <div class="kpi-item">
        <div class="kpi-label">Success Rate</div>
        <div class="kpi-value">{successRate}%</div>
        <div class="kpi-sub">{successRuns}/{totalRuns} runs</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-label">Average Duration</div>
        <div class="kpi-value">{formatDuration(avgDurationMs)}</div>
        <div class="kpi-sub">session-level</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-label">Current Status</div>
        <div class="kpi-value">{running ? 'Running' : done ? 'Completed' : 'Idle'}</div>
        <div class="kpi-sub">task: {selectedTask}</div>
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-header">🧭 Agent Timeline {#if done}<span class="done">✅ Done</span>{/if}</div>
    <div class="timeline-body">
      {#if stageMetrics.length === 0}
        <div class="placeholder">No status yet. Click Run Demo to start.</div>
      {/if}
      {#each stageMetrics as group}
        <div class="stage-card">
          <button class="stage-head" on:click={() => toggleStage(group.stage)}>
            <span class="stage">{group.stage}</span>
            <span class="stage-meta">
              {group.events.length} events · {formatDuration(group.end - group.start)}
              {#if group.warns > 0}<span class="warn-chip">⚠ {group.warns}</span>{/if}
              {#if group.errors > 0}<span class="error-chip">✖ {group.errors}</span>{/if}
            </span>
          </button>

          {#if stageExpanded[group.stage] ?? true}
            <div class="stage-body">
              {#each group.events as item}
                <div class={`timeline-item ${levelClass(item.level)}`}>
                  <span class="time">{toClock(item.timestamp_ms)}</span>
                  <span>{item.message}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  </section>

  <section class="panel">
    <div class="panel-header">📋 Live Logs</div>
    {#if connectionError}
      <div class="error-banner">{connectionError}</div>
    {/if}
    <div class="log-body">
      {#if logs.length === 0}
        <div class="placeholder">Waiting for logs...</div>
      {/if}
      {#each logs as item}
        <div class={`log-line ${levelClass(item.level)}`}>
          <span class="log-tag">[{item.stage} · {toClock(item.timestamp_ms)}]</span> {item.message}
        </div>
      {/each}
    </div>
  </section>

  <section class="panel">
    <div class="panel-header">🚨 Errors</div>
    <div class="error-list">
      {#if errors.length === 0}
        <div class="placeholder">No errors reported.</div>
      {/if}
      {#each errors as item}
        <div class="error-item">[{item.stage} · {toClock(item.timestamp_ms)}] {item.message}</div>
      {/each}
    </div>
  </section>

  {#if summary}
    <section class="panel">
      <div class="panel-header">✅ Result</div>
      <div class="summary">{summary.message}</div>
      {#if summary.meta}
        <pre>{JSON.stringify(summary.meta, null, 2)}</pre>
      {/if}
    </section>
  {/if}
</main>

<style>
  main {
    max-width: 900px;
    margin: 40px auto;
    font-family: sans-serif;
    padding: 0 20px;
    display: grid;
    gap: 16px;
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

  .panel {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
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

  .timeline-body {
    max-height: 220px;
    overflow-y: auto;
    padding: 12px 16px;
    background: #ffffff;
  }

  .timeline-item {
    display: flex;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px dashed #e5e7eb;
    font-size: 14px;
  }

  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    padding: 12px 16px;
    background: #fff;
  }

  .kpi-item {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px;
    background: #f9fafb;
  }

  .kpi-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 4px;
  }

  .kpi-value {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
  }

  .kpi-sub {
    font-size: 12px;
    color: #6b7280;
  }

  .stage-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-bottom: 10px;
    overflow: hidden;
  }

  .stage-head {
    width: 100%;
    border: none;
    background: #f9fafb;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
  }

  .stage-meta {
    color: #4b5563;
    font-size: 12px;
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .stage-body {
    background: #ffffff;
    padding: 0 12px 8px;
  }

  .warn-chip,
  .error-chip {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
  }

  .warn-chip {
    background: #fffbeb;
    color: #b45309;
  }

  .error-chip {
    background: #fef2f2;
    color: #b91c1c;
  }

  .time {
    min-width: 88px;
    color: #6b7280;
    font-size: 12px;
    font-family: monospace;
  }

  .stage {
    min-width: 110px;
    color: #4b5563;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 12px;
  }

  .log-body {
    max-height: 320px;
    overflow-y: auto;
    padding: 12px 16px;
    background: #111827;
  }

  .log-line {
    font-family: monospace;
    font-size: 13px;
    color: #e5e7eb;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .log-tag {
    color: #93c5fd;
  }

  .error-list {
    padding: 12px 16px;
    background: #fff;
  }

  .error-item {
    padding: 8px 10px;
    border-radius: 6px;
    background: #fef2f2;
    color: #b91c1c;
    margin-bottom: 8px;
    font-family: monospace;
    font-size: 12px;
  }

  .summary {
    padding: 12px 16px;
    font-weight: 600;
  }

  pre {
    margin: 0 16px 16px;
    padding: 10px;
    background: #f9fafb;
    border-radius: 6px;
    font-size: 12px;
    overflow: auto;
  }

  .placeholder {
    color: #6b7280;
    font-size: 13px;
  }

  .level-info { color: inherit; }
  .level-warn { color: #f59e0b; }
  .level-error { color: #ef4444; }
</style>
