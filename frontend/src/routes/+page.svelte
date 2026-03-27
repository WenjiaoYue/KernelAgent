<script>
  import { onMount, afterUpdate } from 'svelte';

  // ── State ──────────────────────────────────────────────────
  let models = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'meta-llama/Llama-3.2-1B',
  ];
  let allStages = ['auto_run', 'auto_quant', 'auto_eval'];
  let allDevices = ['xpu', 'cuda'];
  let selectedModel = 'Qwen/Qwen3-0.6B';
  let customModel = '';    // user-typed model ID
  let useCustomModel = false;
  let selectedDevice = 'xpu';
  let selectedStages = new Set(['auto_run', 'auto_quant', 'auto_eval']);

  // ── Machine config (editable) ──────────────────────────────
  let machineHost       = 'h3c.sh.intel.com';
  let machineUser       = 'kaokao';
  let machinePassword   = '123';
  let machineRepoRoot   = '/kaokao/openclaw-triton-gen';
  let machineWorkdir    = '/kaokao/openclaw-triton-gen/examples/auto_run';
  let machineOutputRoot = '/storage/lkk/inference';
  let machineContainer  = 'test_clawd';
  let machineMinimaxKey = 'sk-cp-LKAgYVTb1A-TrGbTtRar7C9HnFtfFxlNdVSTXk2rO9_aBzS5c1JMcZoRCxTmfnP2SHUGQUu3t7G38iIBaswe1p6eV_7hOOZy6KfZ2tVXSCbf2_bl1tZD4gs';

  $: activeModel = useCustomModel ? customModel.trim() : selectedModel;

  // Events
  let timelineEvents = [];   // type === 'status'
  let logsByStage = {};      // stage -> log line[]
  let errorEvents = [];
  let fileEvents = [];       // type === 'file_created'
  let summary = null;

  let running = false;
  let done = false;
  let connectionError = '';
  let hadErrorThisRun = false;
  let debugResult = null;    // result from /api/debug
  let debugLoading = false;
  let runHistory = [];
  let currentRunStartedAt = 0;
  let stageExpanded = {};    // stage -> bool (log accordion)
  let activeLogStage = '';   // which stage log panel is active
  let transcriptByStage = {}; // stage -> transcript message[]
  let activeTranscriptStage = '';

  let transcriptScrollEl = null;  // bind to transcript-body for auto-scroll

  // Workspace file tree: rel_path -> bool (exists)
  const STAGE_FILES = {
    auto_run: [
      'auto_run/transcript.jsonl',
      'auto_run/inference_script.py',
      'auto_run/logs/inference.log',
      'auto_run/summary.md',
    ],
    auto_quant: ['auto_quant/transcript.jsonl'],
    auto_eval:  ['auto_eval/transcript.jsonl'],
  };

  let filePresence = {};
  let stageMetrics = [];
  let totalRuns = 0;
  let successRuns = 0;
  let successRate = 0;
  let avgDurationMs = 0;

  $: filePresence = buildFilePresence(fileEvents);
  $: stageMetrics = buildStageMetrics(timelineEvents, logsByStage, errorEvents);
  $: totalRuns = runHistory.length;
  $: successRuns = runHistory.filter((r) => r.success).length;
  $: successRate = totalRuns === 0 ? 0 : Math.round((successRuns / totalRuns) * 100);
  $: avgDurationMs =
    totalRuns === 0
      ? 0
      : Math.round(runHistory.reduce((s, r) => s + r.durationMs, 0) / totalRuns);

  onMount(async () => {
    // Load stages from backend; models/devices are preset
    const stagesRes = await fetch('http://localhost:8002/api/stages').catch(() => null);
    if (stagesRes?.ok) {
      const sd = await stagesRes.json();
      allStages = sd.stages;
    }
    activeLogStage = allStages[0];
    activeTranscriptStage = allStages[0];
  });

  // Auto-scroll transcript to bottom when new messages arrive
  afterUpdate(() => {
    if (transcriptScrollEl) {
      transcriptScrollEl.scrollTop = transcriptScrollEl.scrollHeight;
    }
  });

  // ── Helpers ────────────────────────────────────────────────
  function parseEvent(raw) {
    try {
      const e = JSON.parse(raw);
      if (!e.timestamp_ms) e.timestamp_ms = Date.now();
      return e;
    } catch {
      return { type: 'log', stage: 'pipeline', level: 'info', message: raw, timestamp_ms: Date.now() };
    }
  }

  function toClock(ms) { return new Date(ms).toLocaleTimeString(); }
  function formatDuration(durationMs) {
    if (!durationMs || durationMs < 1000) {
      return `${durationMs || 0}ms`;
    }
    return `${(durationMs / 1000).toFixed(1)}s`;
  }

  function buildFilePresence(events) {
    const map = {};
    for (const e of events) {
      if (e.meta?.path) map[e.meta.path] = true;
    }
    return map;
  }

  function buildStageMetrics(timeline, logMap, errors) {
    const stageMap = new Map();
    const add = (stage, ts, level, isError) => {
      if (!stageMap.has(stage)) {
        stageMap.set(stage, { stage, start: ts, end: ts, count: 0, warns: 0, errors: 0 });
      }
      const b = stageMap.get(stage);
      b.start = Math.min(b.start, ts);
      b.end   = Math.max(b.end, ts);
      b.count++;
      if (level === 'warn')  b.warns++;
      if (isError)           b.errors++;
    };
    for (const e of timeline) add(e.stage, e.timestamp_ms, e.level, false);
    for (const [stage, lines] of Object.entries(logMap)) {
      for (const e of lines) add(stage, e.timestamp_ms, e.level, e.level === 'error');
    }
    for (const e of errors) add(e.stage, e.timestamp_ms, 'error', true);
    return Array.from(stageMap.values()).sort((a, b) => a.start - b.start);
  }

  function toggleStage(stage) {
    stageExpanded = { ...stageExpanded, [stage]: !(stageExpanded[stage] ?? true) };
  }

  function toggleStageSelect(stage) {
    const next = new Set(selectedStages);
    if (next.has(stage)) { if (next.size > 1) next.delete(stage); }
    else next.add(stage);
    selectedStages = next;
  }

  function selectAll() { selectedStages = new Set(allStages); }

  function levelClass(level) {
    if (level === 'error') return 'level-error';
    if (level === 'warn')  return 'level-warn';
    return 'level-info';
  }

  // ── Run ───────────────────────────────────────────────────
  async function runTask() {
    if (!activeModel || selectedStages.size === 0) return;

    timelineEvents = [];
    logsByStage = {};
    errorEvents = [];
    fileEvents = [];
    transcriptByStage = {};
    summary = null;
    done = false;
    connectionError = '';
    hadErrorThisRun = false;
    currentRunStartedAt = Date.now();
    running = true;
    activeLogStage = [...selectedStages][0];
    activeTranscriptStage = [...selectedStages][0];

    try {
    const res = await fetch('http://localhost:8002/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: activeModel,
        stages: [...selectedStages],
        device: selectedDevice,
        machine: {
          host:        machineHost,
          user:        machineUser,
          password:    machinePassword,
          repo_root:   machineRepoRoot,
          workdir:     machineWorkdir,
          output_root: machineOutputRoot,
          container:   machineContainer,
          minimax_key: machineMinimaxKey,
        },
      }),
    }).catch(err => { throw new Error(`Cannot reach server: ${err.message}`); });
    if (!res.ok) {
      const txt = await res.text().catch(() => res.statusText);
      throw new Error(`Server error ${res.status}: ${txt}`);
    }
    const { task_id } = await res.json();

    const evtSource = new EventSource(`http://localhost:8002/api/stream/${task_id}`);
    evtSource.onmessage = (e) => {
      if (e.data === '[DONE]') {
        evtSource.close();
        running = false;
        done = true;
        runHistory = [
          ...runHistory,
          {
            success: !hadErrorThisRun,
            durationMs: Date.now() - currentRunStartedAt,
            stages: [...selectedStages],
            model: selectedModel,
          },
        ];
        return;
      }

      const event = parseEvent(e.data);
      if (event.type === 'status') {
        timelineEvents = [...timelineEvents, event];
      } else if (event.type === 'error') {
        hadErrorThisRun = true;
        errorEvents = [...errorEvents, event];
      } else if (event.type === 'done') {
        summary = event;
      } else if (event.type === 'file_created') {
        fileEvents = [...fileEvents, event];
      } else if (event.type === 'transcript') {
        // Route 'pipeline' stage events to the first selected stage tab
        // (backend uses 'pipeline' as stage label for multi-stage runs)
        const raw = event.stage || '';
        const s = (raw && raw !== 'pipeline' && selectedStages.has(raw))
          ? raw
          : ([...selectedStages][0] || 'pipeline');
        transcriptByStage = { ...transcriptByStage, [s]: [...(transcriptByStage[s] || []), event] };
        activeTranscriptStage = s;
      } else {
        const s = event.stage || 'pipeline';
        logsByStage = { ...logsByStage, [s]: [...(logsByStage[s] || []), event] };
        if (event.level === 'error') {
          hadErrorThisRun = true;
          errorEvents = [...errorEvents, event];
        }
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      running = false;
      hadErrorThisRun = true;
      connectionError = 'Stream connection failed. Please try again.';
    };
    } catch (err) {
      running = false;
      connectionError = String(err.message || err);
    }
  }

  // ── Debug check ───────────────────────────────────────────
  async function checkDebug() {
    debugLoading = true;
    debugResult = null;
    try {
      const res = await fetch('http://localhost:8002/api/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host:        machineHost,
          user:        machineUser,
          password:    machinePassword,
          container:   machineContainer,
          output_root: machineOutputRoot,
          model:       activeModel,
          stages:      [...selectedStages],
        }),
      });
      debugResult = await res.json();
    } catch (err) {
      debugResult = { ssh: `Cannot reach backend: ${err.message}` };
    } finally {
      debugLoading = false;
    }
  }
</script>

<main>
  <h1>🤖 KernelAgent Pipeline Console</h1>

  <!-- Controls -->
  <section class="panel">
    <div class="panel-header">⚙️ Configuration</div>
    <div class="controls">
      <!-- Model: text input + preset chips -->
      <div class="model-row">
        <span class="ctrl-label">Model</span>
        <div class="model-input-group">
          <input
            class="model-input"
            type="text"
            placeholder="e.g. Qwen/Qwen3-0.6B"
            bind:value={selectedModel}
            on:focus={() => (useCustomModel = false)}
          />
        </div>
      </div>
      <div class="model-presets">
        {#each models as m}
          <button
            class="chip {selectedModel === m && !useCustomModel ? 'chip-active' : 'chip-inactive'}"
            on:click={() => { selectedModel = m; useCustomModel = false; }}
          >{m.split('/')[1]}</button>
        {/each}
      </div>

      <div class="stage-selector">
        <span class="ctrl-label">Device</span>
        <div class="stage-chips">
          {#each allDevices as dev}
            <button
              class="chip {selectedDevice === dev ? 'chip-active' : 'chip-inactive'}"
              on:click={() => (selectedDevice = dev)}
            >
              {dev.toUpperCase()}
            </button>
          {/each}
        </div>
      </div>

      <div class="stage-selector">
        <span class="ctrl-label">Stages</span>
        <div class="stage-chips">
          {#each allStages as stage}
            <button
              class="chip {selectedStages.has(stage) ? 'chip-active' : 'chip-inactive'}"
              on:click={() => toggleStageSelect(stage)}
            >
              {stage}
            </button>
          {/each}
          <button class="chip chip-util" on:click={selectAll}>All</button>
        </div>
      </div>

      <div class="buttons">
        <button class="btn-run" disabled={running} on:click={runTask}>
          {running ? '⏳ Running…' : '▶ Run Pipeline'}
        </button>
        <button class="btn-debug" disabled={debugLoading} on:click={checkDebug}>
          {debugLoading ? '⏳ Checking…' : '🔍 Debug Connection'}
        </button>
      </div>
    </div>
    {#if connectionError}
      <div class="error-banner">{connectionError}</div>
    {/if}
  </section>

  <!-- Machine Config (collapsible) -->
  <details class="panel collapsible-panel">
    <summary class="panel-header">🖥️ Machine Config <span class="panel-sub">{machineHost} · {machineContainer}</span></summary>
    <div class="machine-grid">
      <div class="mfield">
        <span class="mfield-label">SSH Host</span>
        <input class="mfield-input" bind:value={machineHost} placeholder="hostname or IP" />
      </div>
      <div class="mfield">
        <span class="mfield-label">SSH User</span>
        <input class="mfield-input" bind:value={machineUser} placeholder="username" />
      </div>
      <div class="mfield">
        <span class="mfield-label">SSH Password</span>
        <input class="mfield-input" type="password" bind:value={machinePassword} placeholder="password" />
      </div>
      <div class="mfield">
        <span class="mfield-label">Docker Container</span>
        <input class="mfield-input" bind:value={machineContainer} placeholder="container name" />
      </div>
      <div class="mfield mfield-wide">
        <span class="mfield-label">Repo Root</span>
        <input class="mfield-input" bind:value={machineRepoRoot} placeholder="/path/to/repo" />
      </div>
      <div class="mfield mfield-wide">
        <span class="mfield-label">Script Workdir</span>
        <input class="mfield-input" bind:value={machineWorkdir} placeholder="/path/to/workdir" />
      </div>
      <div class="mfield mfield-wide">
        <span class="mfield-label">Output Root</span>
        <input class="mfield-input" bind:value={machineOutputRoot} placeholder="/path/to/output" />
      </div>
      <div class="mfield mfield-wide">
        <span class="mfield-label">MiniMax API Key</span>
        <input class="mfield-input" type="password" bind:value={machineMinimaxKey} placeholder="sk-..." />
      </div>
    </div>
  </details>

  <!-- KPIs -->
  <section class="panel">
    <div class="panel-header">📊 Session KPIs</div>
    <div class="kpi-grid">
      <div class="kpi-item">
        <div class="kpi-label">Success Rate</div>
        <div class="kpi-value">{successRate}%</div>
        <div class="kpi-sub">{successRuns}/{totalRuns} runs</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-label">Avg Duration</div>
        <div class="kpi-value">{formatDuration(avgDurationMs)}</div>
        <div class="kpi-sub">session-level</div>
      </div>
      <div class="kpi-item">
        <div class="kpi-label">Current Status</div>
        <div class="kpi-value {running ? 'status-running' : done ? 'status-done' : 'status-idle'}">
          {running ? 'Running' : done ? 'Done ✅' : 'Idle'}
        </div>
        <div class="kpi-sub">{[...selectedStages].join(' + ')} · {selectedDevice.toUpperCase()}</div>
      </div>
    </div>
  </section>

  <!-- Timeline -->
  <section class="panel">
    <div class="panel-header">
      🧭 Pipeline Timeline
      {#if done}<span class="done-badge">✅ Done</span>{/if}
    </div>
    <div class="timeline-body">
      {#if stageMetrics.length === 0}
        {#if running}
          <div class="panel-loading"><span class="spinner"></span> Waiting for pipeline events…</div>
        {:else}
          <div class="placeholder">No status yet. Click Run Pipeline to start.</div>
        {/if}
      {/if}
      {#each stageMetrics as group}
        <div class="stage-card">
          <button class="stage-head" on:click={() => toggleStage(group.stage)}>
            <span class="stage-pill">{group.stage}</span>
            <span class="stage-meta">
              {group.count} events · {formatDuration(group.end - group.start)}
              {#if group.warns > 0}<span class="warn-chip">⚠ {group.warns}</span>{/if}
              {#if group.errors > 0}<span class="error-chip">✖ {group.errors}</span>{/if}
            </span>
          </button>
          {#if stageExpanded[group.stage] ?? true}
            <div class="stage-body">
              {#each timelineEvents.filter(e => e.stage === group.stage) as item}
                <div class="timeline-item {levelClass(item.level)}">
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

  <!-- Agent Transcript — below Timeline -->
  <section class="panel panel-transcript">
    <div class="panel-header">
      💬 Agent Activity
      <span class="panel-sub">
        {running ? '🟢 live' : '—'}
        {#if running}· session: autorun_{activeModel.replace('/', '_')}{/if}
      </span>
    </div>

    <div class="log-tabs">
      {#each allStages as stage}
        {#if selectedStages.has(stage)}
          <button
            class="log-tab {activeTranscriptStage === stage ? 'log-tab-active' : ''}"
            on:click={() => (activeTranscriptStage = stage)}
          >
            {stage}
            {#if (transcriptByStage[stage] || []).length > 0}
              <span class="log-count">{(transcriptByStage[stage] || []).length}</span>
            {/if}
          </button>
        {/if}
      {/each}
    </div>

    <div class="transcript-body" bind:this={transcriptScrollEl}>
      {#each (transcriptByStage[activeTranscriptStage] || []) as msg, i}
        {@const sub = msg.meta?.sub_type || 'text'}
        {@const toolName = msg.meta?.tool_name || 'tool'}
        {@const toolIcon = toolName === 'exec' ? '🖥️' : toolName === 'read' ? '📖' : toolName === 'write' ? '✍️' : toolName === 'web_fetch' ? '🌐' : '🔧'}
        {@const lineCount = (msg.message || '').split('\n').length}

        {#if sub === 'thinking'}
          <details class="tx-thinking">
            <summary>
              <span class="tx-flow">
                <span class="tx-badge tx-badge-assistant">assistant</span>
                <span class="tx-arrow">→</span>
                <span class="tx-badge tx-badge-llm">LLM</span>
              </span>
              <span class="tx-thinking-label">💭 thinking</span>
            </summary>
            <pre class="tx-thinking-body">{msg.message}</pre>
          </details>

        {:else if sub === 'tool_call'}
          <div class="tx-action">
            <div class="tx-action-header">
              <span class="tx-flow">
                <span class="tx-badge tx-badge-agent">agent</span>
                <span class="tx-arrow">→</span>
                <span class="tx-badge tx-badge-tool">tool</span>
              </span>
              <span class="tx-icon">{toolIcon}</span>
              <span class="tx-tool-name">{toolName.toUpperCase()}</span>
              <span class="tx-step">#{i + 1}</span>
            </div>
            <pre class="tx-cmd">{msg.message}</pre>
          </div>

        {:else if sub === 'tool_result'}
          {@const readPath = msg.meta?.read_path || ''}
          <details class="tx-result">
            <summary>
              <span class="tx-flow">
                <span class="tx-badge tx-badge-tool">tool</span>
                <span class="tx-arrow">→</span>
                <span class="tx-badge tx-badge-agent">agent</span>
              </span>
              <span class="tx-result-label">📤 output</span>
              {#if readPath}
                <span class="tx-result-path">📄 {readPath}</span>
              {/if}
              <span class="tx-result-lines">{lineCount} {lineCount === 1 ? 'line' : 'lines'}</span>
            </summary>
            <pre class="tx-result-body">{msg.message}</pre>
          </details>

        {:else}
          <div class="tx-text">
            <div class="tx-text-header">
              <span class="tx-badge tx-badge-assistant">assistant</span>
              <span class="tx-text-icon">🤖</span>
            </div>
            <div class="tx-text-body">{msg.message}</div>
          </div>
        {/if}

      {/each}
      {#if (transcriptByStage[activeTranscriptStage] || []).length === 0}
        {#if running}
          <div class="tx-empty"><span class="spinner"></span> Waiting for openclaw session…</div>
        {:else if done}
          <div class="tx-empty">No agent activity recorded for this stage.</div>
        {/if}
      {/if}
    </div>
  </section>

  <!-- Per-Stage Live Logs (process stdout) -->
  <details class="panel collapsible-panel">
    <summary class="panel-header">🖥️ Process Logs (stdout/stderr)</summary>

    <div class="log-tabs">
      {#each allStages as stage}
        {#if selectedStages.has(stage)}
          <button
            class="log-tab {activeLogStage === stage ? 'log-tab-active' : ''}"
            on:click={() => (activeLogStage = stage)}
          >
            {stage}
            {#if (logsByStage[stage] || []).length > 0}
              <span class="log-count">{(logsByStage[stage] || []).length}</span>
            {/if}
          </button>
        {/if}
      {/each}
      {#if logsByStage['pipeline']}
        <button
          class="log-tab {activeLogStage === 'pipeline' ? 'log-tab-active' : ''}"
          on:click={() => (activeLogStage = 'pipeline')}
        >
          pipeline
          <span class="log-count">{(logsByStage['pipeline'] || []).length}</span>
        </button>
      {/if}
    </div>

    <div class="log-body">
      {#each (logsByStage[activeLogStage] || []) as line}
        <div class="log-line {levelClass(line.level)}">
          <span class="log-time">{toClock(line.timestamp_ms)}</span>
          {line.message}
        </div>
      {/each}
      {#if (logsByStage[activeLogStage] || []).length === 0}
        {#if running}
          <div class="placeholder log-placeholder"><span class="spinner spinner-light"></span> Waiting for output…</div>
        {:else if !done}
          <div class="placeholder log-placeholder">No log lines yet.</div>
        {/if}
      {/if}
    </div>
  </details>

  <!-- Errors -->
  {#if errorEvents.length > 0}
    <section class="panel">
      <div class="panel-header">🚨 Errors ({errorEvents.length})</div>
      <div class="error-list">
        {#each errorEvents as item}
          <div class="error-item">
            <span class="error-stage">[{item.stage}]</span>
            <span class="error-time">{toClock(item.timestamp_ms)}</span>
            {item.message}
          </div>
        {/each}
      </div>
    </section>
  {/if}

  <!-- Summary -->
  {#if summary}
    <section class="panel">
      <div class="panel-header">✅ Result</div>
      <div class="summary">{summary.message}</div>
      {#if summary.meta}
        <pre>{JSON.stringify(summary.meta, null, 2)}</pre>
      {/if}
    </section>
  {/if}

  <!-- Debug panel -->
  {#if debugResult}
    <section class="panel panel-debug">
      <div class="panel-header">🔍 Connection Diagnostics</div>
      <div class="debug-body">
        <div class="debug-row">
          <span class="debug-label">SSH</span>
          <span class="debug-val {debugResult.ssh === 'ok' ? 'debug-ok' : 'debug-fail'}">{debugResult.ssh}</span>
        </div>
        <div class="debug-row">
          <span class="debug-label">Session key</span>
          <code class="debug-code">{debugResult.session_key}</code>
        </div>
        <div class="debug-row">
          <span class="debug-label">Resolved file</span>
          <code class="debug-code {debugResult.resolved_session_file ? 'debug-ok' : 'debug-fail'}">
            {debugResult.resolved_session_file || '— not found —'}
          </code>
        </div>
        {#if debugResult.session_file_lines}
          <div class="debug-row">
            <span class="debug-label">Session lines</span>
            <span class="debug-val">{debugResult.session_file_lines}</span>
          </div>
        {/if}
        <div class="debug-section-label">Transcript files in output dir</div>
        {#each Object.entries(debugResult.transcript_paths || {}) as [path, stat]}
          <div class="debug-row debug-row-file">
            <span class="debug-file-stat {stat === 'NOT_FOUND' ? 'debug-fail' : 'debug-ok'}">{stat === 'NOT_FOUND' ? '✖' : '✔'}</span>
            <code class="debug-code">{path}</code>
            <span class="debug-file-lines">{stat}</span>
          </div>
        {/each}
        <div class="debug-section-label">Sessions in .openclaw ({(debugResult.session_files || []).length} files)</div>
        {#each (debugResult.session_files || []) as f}
          <div class="debug-file-item"><code>{f}</code></div>
        {/each}
        {#if (debugResult.sessions_json_keys || []).length}
          <div class="debug-section-label">sessions.json keys</div>
          {#each (debugResult.sessions_json_keys || []) as k}
            <div class="debug-file-item"><code>{k}</code></div>
          {/each}
        {/if}
      </div>
    </section>
  {/if}
</main>

<style>
  main {
    max-width: 960px;
    margin: 32px auto;
    font-family: 'Inter', system-ui, sans-serif;
    padding: 0 20px;
    display: grid;
    gap: 16px;
  }

  h1 { margin-bottom: 8px; font-size: 22px; }

  /* ── Panel ── */
  .panel {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
  }

  .panel-transcript {
    border-color: #3b82f6;
    box-shadow: 0 0 0 1px #3b82f633;
  }

  .collapsible-panel {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
  }

  /* ── Machine Config grid ── */
  .machine-grid {
    padding: 14px 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 16px;
  }
  .mfield-wide { grid-column: 1 / -1; }
  .mfield { display: flex; flex-direction: column; gap: 4px; }
  .mfield-label {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .mfield-input {
    padding: 6px 10px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 13px;
    font-family: 'Fira Mono', monospace;
    box-sizing: border-box;
    width: 100%;
    background: #f9fafb;
  }
  .mfield-input:focus { outline: 2px solid #3b82f6; border-color: #3b82f6; background: #fff; }

  /* ── Loading spinner ── */
  .panel-loading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #6b7280;
    font-size: 13px;
    padding: 14px 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid #d1d5db;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }
  .spinner-light {
    border-color: #334155;
    border-top-color: #7dd3fc;
  }
  .collapsible-panel > summary {
    cursor: pointer;
    user-select: none;
    list-style: none;
  }
  .collapsible-panel > summary::-webkit-details-marker { display: none; }

  .panel-header {
    background: #f9fafb;
    padding: 10px 16px;
    font-weight: 600;
    font-size: 14px;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* ── Controls ── */
  .controls {
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .ctrl-label {
    min-width: 72px;
    font-weight: 600;
    font-size: 13px;
    color: #374151;
  }

  /* Model input */
  .model-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .model-input-group { flex: 1; }
  .model-input {
    width: 100%;
    padding: 6px 10px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
    box-sizing: border-box;
  }
  .model-input:focus { outline: 2px solid #3b82f6; border-color: #3b82f6; }
  .model-presets {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-left: 84px;
  }

  .stage-selector {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .stage-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .chip {
    border: none;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    font-weight: 500;
  }

  .chip-active   { background: #3b82f6; color: #fff; }
  .chip-inactive { background: #e5e7eb; color: #6b7280; }
  .chip-inactive:hover { background: #d1d5db; }
  .chip-util { background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; }
  .chip-util:hover { background: #e5e7eb; }

  .buttons { display: flex; gap: 10px; }

  .btn-run {
    padding: 8px 20px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: background 0.2s;
  }
  .btn-run:disabled { background: #93c5fd; cursor: not-allowed; }
  .btn-run:hover:not(:disabled) { background: #2563eb; }

  .error-banner {
    background: #fef2f2;
    color: #dc2626;
    padding: 8px 16px;
    border-top: 1px solid #fecaca;
    font-size: 13px;
  }

  /* ── KPIs ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    padding: 14px 16px;
  }

  .kpi-item {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    background: #f9fafb;
  }

  .kpi-label { font-size: 11px; color: #6b7280; margin-bottom: 4px; text-transform: uppercase; }
  .kpi-value { font-size: 22px; font-weight: 700; color: #111827; }
  .kpi-sub   { font-size: 11px; color: #6b7280; margin-top: 2px; }

  .status-running { color: #d97706; }
  .status-done    { color: #16a34a; }
  .status-idle    { color: #6b7280; }
  .done-badge { color: #16a34a; font-weight: 600; font-size: 13px; }

  /* ── Timeline ── */
  .timeline-body {
    max-height: 280px;
    overflow-y: auto;
    padding: 12px 16px;
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
    background: #f3f4f6;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    text-align: left;
  }

  .stage-pill {
    background: #dbeafe;
    color: #1e40af;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
  }

  .stage-meta {
    color: #4b5563;
    font-size: 12px;
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .stage-body { padding: 6px 12px 10px; background: #fff; }

  .timeline-item {
    display: flex;
    gap: 10px;
    padding: 4px 0;
    font-size: 13px;
    border-bottom: 1px dashed #e5e7eb;
  }

  .time {
    min-width: 82px;
    color: #9ca3af;
    font-size: 11px;
    font-family: monospace;
    margin-top: 2px;
  }

  .warn-chip, .error-chip {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
  }
  .warn-chip  { background: #fffbeb; color: #b45309; }
  .error-chip { background: #fef2f2; color: #b91c1c; }

  /* ── File Tree ── */
  .file-tree {
    padding: 14px 16px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
  }

  .tree-stage-label {
    font-size: 13px;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
  }

  .tree-file {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-family: monospace;
    margin-bottom: 4px;
  }

  .file-exists  { background: #f0fdf4; color: #15803d; }
  .file-pending { background: #f9fafb; color: #9ca3af; }
  .file-icon { font-size: 14px; }
  .file-name { flex: 1; }

  /* ── Logs ── */
  .log-tabs {
    display: flex;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
    overflow-x: auto;
  }

  .log-tab {
    padding: 8px 16px;
    border: none;
    border-bottom: 3px solid transparent;
    background: transparent;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #6b7280;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: color 0.15s;
  }
  .log-tab:hover { color: #111827; }
  .log-tab-active { color: #3b82f6; border-bottom-color: #3b82f6; background: #fff; }

  .log-count {
    background: #e5e7eb;
    border-radius: 999px;
    padding: 1px 7px;
    font-size: 11px;
    color: #374151;
  }
  .log-tab-active .log-count { background: #dbeafe; color: #1e40af; }

  .log-body {
    max-height: 380px;
    overflow-y: auto;
    padding: 10px 14px;
    background: #0f172a;
    color: #cbd5e1;
  }

  .log-line {
    font-family: 'Fira Mono', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-all;
    color: #cbd5e1;
  }

  .log-time { color: #64748b; margin-right: 8px; font-size: 11px; }
  :global(.log-line.level-warn)  { color: #fbbf24; }
  :global(.log-line.level-error) { color: #f87171; }
  .log-placeholder { color: #94a3b8; font-family: monospace; font-size: 12px; }

  /* ── Transcript (Activity Feed) ── */
  .panel-sub {
    font-size: 11px;
    font-weight: 400;
    color: #9ca3af;
  }

  .transcript-body {
    max-height: 600px;
    overflow-y: auto;
    padding: 12px 16px;
    background: #0f172a;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .tx-empty {
    color: #475569;
    font-size: 13px;
    padding: 20px 0;
    text-align: center;
  }

  /* ── Flow badges ── */
  .tx-flow {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-right: 8px;
  }
  .tx-arrow { color: #475569; font-size: 10px; }
  .tx-badge {
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 1px 5px;
    border-radius: 3px;
    text-transform: uppercase;
    font-family: monospace;
  }
  .tx-badge-assistant { background: #312e81; color: #a5b4fc; border: 1px solid #4338ca; }
  .tx-badge-llm       { background: #1e3a5f; color: #7dd3fc; border: 1px solid #0369a1; }
  .tx-badge-agent     { background: #14532d; color: #86efac; border: 1px solid #16a34a; }
  .tx-badge-tool      { background: #431407; color: #fdba74; border: 1px solid #c2410c; }

  /* Thinking — collapsed, subtle */
  .tx-thinking {
    border-left: 2px solid #312e81;
    padding-left: 8px;
    border-radius: 0 4px 4px 0;
  }
  .tx-thinking summary {
    cursor: pointer;
    color: #64748b;
    font-size: 11px;
    padding: 4px 0;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .tx-thinking-label { color: #6366f1; font-style: italic; }
  .tx-thinking-body {
    margin: 4px 0 0;
    padding: 8px 12px;
    background: #0f0f1a;
    font-size: 11px;
    color: #818cf8;
    font-style: italic;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: inherit;
    border-left: 1px dashed #312e81;
    border-radius: 0 4px 4px 0;
    line-height: 1.6;
  }

  /* Tool Call — bright action card */
  .tx-action {
    background: #0d1b2a;
    border: 1px solid #0369a1;
    border-radius: 6px;
    overflow: hidden;
  }
  .tx-action-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: #0a1628;
    border-bottom: 1px solid #0c2a45;
  }
  .tx-icon { font-size: 12px; }
  .tx-tool-name {
    font-size: 11px;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    flex: 1;
  }
  .tx-step {
    font-size: 10px;
    color: #334155;
    font-family: monospace;
  }
  .tx-cmd {
    margin: 0;
    padding: 8px 12px;
    background: #0d1b2a;
    color: #e2e8f0;
    font-family: 'Fira Mono', 'Consolas', monospace;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.6;
  }

  /* Tool Result — collapsible output panel */
  .tx-result {
    border: 1px solid #431407;
    border-radius: 6px;
    overflow: hidden;
    margin-left: 16px;
  }
  .tx-result summary {
    cursor: pointer;
    background: #1c0a00;
    padding: 5px 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    user-select: none;
    list-style: none;
  }
  .tx-result-label { color: #fb923c; font-size: 11px; font-weight: 600; }
  .tx-result-path {
    color: #94a3b8;
    font-size: 10px;
    font-family: 'Fira Mono', monospace;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .tx-result-lines { color: #57534e; font-size: 10px; font-family: monospace; margin-left: auto; flex-shrink: 0; }
  .tx-result-body {
    margin: 0;
    padding: 10px 12px;
    background: #0a0a0a;
    color: #86efac;
    font-family: 'Fira Mono', 'Consolas', monospace;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.6;
  }

  /* Assistant text */
  .tx-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 6px 0;
    border-left: 2px solid #312e81;
    padding-left: 10px;
  }
  .tx-text-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tx-text-icon { font-size: 13px; }
  .tx-text-body {
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Errors ── */
  .error-list { padding: 12px 16px; }

  .error-item {
    padding: 8px 12px;
    border-radius: 6px;
    background: #fef2f2;
    color: #991b1b;
    margin-bottom: 8px;
    font-family: monospace;
    font-size: 12px;
    line-height: 1.5;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .error-stage { font-weight: 700; color: #b91c1c; }
  .error-time  { color: #9ca3af; }

  /* ── Summary ── */
  .summary { padding: 12px 16px; font-weight: 600; color: #15803d; }

  pre {
    margin: 0 16px 16px;
    padding: 10px;
    background: #f9fafb;
    border-radius: 6px;
    font-size: 12px;
    overflow: auto;
  }

  /* ── Misc ── */
  .placeholder { color: #9ca3af; font-size: 13px; }
  .level-info  { color: inherit; }
  .level-warn  { color: #f59e0b; }
  .level-error { color: #ef4444; }

  /* ── Debug button ── */
  .btn-debug {
    padding: 8px 16px;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: background 0.15s;
  }
  .btn-debug:hover:not(:disabled) { background: #e5e7eb; }
  .btn-debug:disabled { opacity: 0.6; cursor: not-allowed; }

  /* ── Debug panel ── */
  .panel-debug { border-color: #6366f1; }
  .debug-body { padding: 12px 16px; display: flex; flex-direction: column; gap: 6px; }
  .debug-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: 13px;
  }
  .debug-row-file { align-items: center; }
  .debug-label {
    min-width: 120px;
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .debug-val { font-size: 13px; color: #111827; }
  .debug-code {
    font-family: 'Fira Mono', monospace;
    font-size: 12px;
    color: #374151;
    background: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
    flex: 1;
    word-break: break-all;
  }
  .debug-ok   { color: #15803d !important; }
  .debug-fail { color: #b91c1c !important; }
  .debug-section-label {
    font-size: 11px;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    margin-top: 8px;
    border-top: 1px solid #e5e7eb;
    padding-top: 6px;
  }
  .debug-file-item { font-size: 12px; font-family: monospace; color: #4b5563; padding-left: 4px; }
  .debug-file-stat { font-size: 13px; min-width: 16px; }
  .debug-file-lines { font-size: 11px; color: #9ca3af; font-family: monospace; margin-left: auto; }
</style>
    