<script lang="ts">
  import { onDestroy } from 'svelte';

  type Tab = 'zimage' | 'excelxlx' | 'pipeline' | 'hfdiscover';
  type LogItem = {
    timestamp: string;
    type: string;
    tag?: string;
    message: string;
  };

  // ─── Per-tab state ──────────────────────────────────────────────────────
  type TabState = {
    sessionLog: LogItem[];
    taskLog: LogItem[];
    sessionUserScrolledUp: boolean;
    taskUserScrolledUp: boolean;
    collapsedStatus: Set<number>;
  };
  const tabStates: Record<Tab, TabState> = {
    zimage:      { sessionLog: [], taskLog: [], sessionUserScrolledUp: false, taskUserScrolledUp: false, collapsedStatus: new Set() },
    excelxlx:    { sessionLog: [], taskLog: [], sessionUserScrolledUp: false, taskUserScrolledUp: false, collapsedStatus: new Set() },
    pipeline:    { sessionLog: [], taskLog: [], sessionUserScrolledUp: false, taskUserScrolledUp: false, collapsedStatus: new Set() },
    hfdiscover:  { sessionLog: [], taskLog: [], sessionUserScrolledUp: false, taskUserScrolledUp: false, collapsedStatus: new Set() },
  };

  function toggleCollapsedStatus(idx: number) {
    const s = tabStates[currentTab].collapsedStatus;
    if (s.has(idx)) { s.delete(idx); } else { s.add(idx); }
    tabStates[currentTab].collapsedStatus = new Set(s);
  }

  let currentTab: Tab = 'zimage';
  // These are reactive references into the current tab's state
  $: sessionLog = tabStates[currentTab].sessionLog;
  $: taskLog = tabStates[currentTab].taskLog;

  // Active-tab helpers that write into the correct tab's state
  let activeTab: Tab = 'zimage';  // tracks which tab owns the running task

  function appendSessionLog(item: LogItem) {
    tabStates[activeTab].sessionLog = [...tabStates[activeTab].sessionLog, item];
  }
  function appendTaskLog(item: LogItem) {
    tabStates[activeTab].taskLog = [...tabStates[activeTab].taskLog, item];
  }

  let evtSource: EventSource | null = null;
  let isRunning = false;
  let showMachineConfig = false;
  let zimageImgSrcMap: Record<Tab, string> = { zimage: '', excelxlx: '', pipeline: '', hfdiscover: '' };
  $: zimageImgSrc = zimageImgSrcMap[currentTab];

  // HF Discover form
  let hfDays = 10;
  let hfLimit = 20;
  let hfSort = 'createdAt';
  let hfPipelineTag = '';
  let hfTimeout = 120;
  let hfSession = 'hf_discover';

  // HF Discover results
  type HFModel = { id: string; pipeline_tag: string; likes: number; downloads: number; createdAt: string; selected?: boolean };
  let hfModels: HFModel[] = [];
  let hfSelectAll = false;
  let hfLoading = false;
  let hfError = '';

  const BACKEND = window.location.origin;

  // Z-Image form
  let ziPrompt = 'a cup of coffee on the wooden table, photorealistic, 8k';
  let ziDevice = 'xpu';
  let ziSteps = 9;
  let ziSeed = 42;
  let ziOutput = '/storage/lkk/inference/zimage/zimage_output.png';
  let ziSession = 'zimage_task';

  // Excel-XLX form
  let xlxFilePath = '/tmp/agent_profiling_swe_2core.xlsx';
  let xlxInstructions = 'Read all sheet names and the first 5 rows of each sheet, display in table format';
  let xlxSkill = '/wenjiao/openclaw-triton-gen/skills/excel-xlsx/SKILL.md';
  let xlxSession = 'excel_xlx_task';
  let xlxTimeout = 300;  let xlxFileContent = '';   // rendered markdown from /api/read-xlsx
  // Pipeline form
  let plModel = 'Qwen/Qwen3-0.6B';
  let plDevice = 'xpu';
  let plDeviceIndex = 0;
  let plTimeout = 7200;
  let plSession = 'pipeline';
  let plChkTest = false;
  let plChkQuant = true;
  let plChkEval = true;
  let plTestOutput = '/storage/lkk/inference';
  let plAutorunSkill = '/root/.openclaw/workspace/skills/auto_run/SKILL.md';
  let plScheme = 'W4A16';
  let plMethod = 'RTN';
  let plExport = 'auto_round:auto_gptq';
  let plQuantOutput = '/kaokao/quantized';
  let plAutoquantSkill = '/root/.openclaw/workspace/skills/auto_quant/SKILL.md';
  let plTasks = 'piqa';
  let plEvalOutput = '/kaokao/lm_eval_results';
  let plBatchSize = 8;
  let plMaxLen = 8192;
  let plGpuMem = 0.8;
  let plSkill = '/root/.openclaw/workspace/skills/auto_eval/SKILL.md';

  // Machine Config
  let mcHost = 'h3c.sh.intel.com';
  let mcUser = 'kaokao';
  let mcPassword = '';

  let mcContainer = 'xpu-openclaw';
  let mcRepoRoot = '/wenjiao/openclaw-triton-gen';
  let mcWorkdir = '/wenjiao/openclaw-triton-gen/examples/auto_run';
  let mcOutputRoot = '/storage/lkk/inference';
  let mcSessionDir = '/root/.openclaw/agents/main/sessions';
  let mcMinimaxKey = '';

  $: plEvalInput = plModel && plQuantOutput
    ? `${plQuantOutput}/${plModel.replace(/\//g, '_')}-${plScheme}`
    : '';

  function ts(): string {
    const d = new Date();
    return d.toTimeString().slice(0, 8);
  }

  function getMachineConfig() {
    return {
      host: mcHost || 'h3c.sh.intel.com',
      user: mcUser || 'kaokao',
      password: mcPassword,
      container: mcContainer || 'test_clawd',
      repo_root: mcRepoRoot || '/kaokao/openclaw-triton-gen',
      workdir: mcWorkdir || '/kaokao/openclaw-triton-gen/examples/auto_run',
      output_root: mcOutputRoot || '/storage/lkk/inference',
      session_dir: mcSessionDir || '/root/.openclaw/agents/main/sessions',
      minimax_key: mcMinimaxKey,
    };
  }

  function handleEvent(e: MessageEvent) {
    if (e.data === '[DONE]') {
      isRunning = false;
      return;
    }

    let obj: any;
    try {
      obj = JSON.parse(e.data);
    } catch {
      return;
    }

    const type = obj.type || '';
    const msg = obj.message || '';
    const sub = (obj.meta || {}).sub_type || '';

    if (type === 'zimage_log') {
      appendTaskLog({ timestamp: ts(), type: 'zimage_log', message: msg });
      return;
    }

    if (type === 'transcript') {
      const tagMap: Record<string, string> = { thinking: 'Think', tool_call: 'Tool', tool_result: 'Result', text: 'Text' };
      appendSessionLog({
        timestamp: ts(),
        type: sub || 'text',
        tag: tagMap[sub] || sub,
        message: msg
      });
      // Parse model list as soon as transcript contains JSON data (don't wait for done)
      if (activeTab === 'hfdiscover' && hfModels.length === 0 && (sub === 'text' || sub === 'tool_result')) {
        parseModelsFromLog();
      }
      return;
    }

    if (type === 'status') {
      appendSessionLog({ timestamp: ts(), type: 'status', message: msg });
      return;
    }

    if (type === 'log') {
      appendTaskLog({ timestamp: ts(), type: 'log', message: msg });
      return;
    }

    if (type === 'done') {
      appendSessionLog({ timestamp: ts(), type: 'done', message: msg });
      if (activeTab === 'zimage' && obj.meta && obj.meta.output) {
        const mc = getMachineConfig();
        zimageImgSrcMap['zimage'] = `${BACKEND}/api/image?path=${encodeURIComponent(obj.meta.output)}&container=${encodeURIComponent(mc.container)}`;
        zimageImgSrcMap = { ...zimageImgSrcMap };
      }
      if (activeTab === 'hfdiscover') {
        setTimeout(() => parseModelsFromLog(), 100);
      }
      return;
    }

    if (type === 'error') {
      appendSessionLog({ timestamp: ts(), type: 'error', message: msg });
      return;
    }
  }

  function subscribeStream(task_id: string) {
    if (evtSource) evtSource.close();
    evtSource = new EventSource(`${BACKEND}/api/stream/${task_id}`);
    evtSource.onmessage = handleEvent;
    evtSource.onerror = () => {
      isRunning = false;
      if (evtSource) evtSource.close();
    };
  }

  async function postAndStream(endpoint: string, body: any) {
    activeTab = currentTab;
    tabStates[activeTab].sessionLog = [];
    tabStates[activeTab].taskLog = [];
    tabStates[activeTab].sessionUserScrolledUp = false;
    tabStates[activeTab].taskUserScrolledUp = false;
    isRunning = true;

    let res: Response;
    try {
      res = await fetch(`${BACKEND}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err: any) {
      appendSessionLog({ timestamp: ts(), type: 'error', message: 'Connection failed: ' + err.message });
      isRunning = false;
      return;
    }

    if (!res.ok) {
      const t = await res.text();
      appendSessionLog({ timestamp: ts(), type: 'error', message: `HTTP ${res.status}: ${t}` });
      isRunning = false;
      return;
    }

    const { task_id } = await res.json();
    subscribeStream(task_id);
  }

  function runZImage() {
    if (!ziPrompt.trim()) {
      alert('Please enter a Prompt');
      return;
    }
    postAndStream('/api/run-zimage', {
      prompt: ziPrompt,
      output: ziOutput,
      num_inference_steps: ziSteps,
      seed: ziSeed,
      device: ziDevice,
      session_key: ziSession || 'zimage_task',
      machine: getMachineConfig(),
    });
  }

  function runExcelXLX() {
    if (!xlxInstructions.trim()) {
      alert('Please enter instructions');
      return;
    }
    postAndStream('/api/run-excel-xlx', {
      file_path: xlxFilePath,
      instructions: xlxInstructions,
      skill_path: xlxSkill,
      session_key: xlxSession || 'excel_xlx_task',
      timeout: xlxTimeout,
      machine: getMachineConfig(),
    });
  }

  function runPipeline() {
    if (!plModel.trim()) {
      alert('Please enter a Model ID');
      return;
    }
    if (!plChkTest && !plChkQuant && !plChkEval) {
      alert('Please select at least one stage');
      return;
    }
    postAndStream('/api/run-pipeline', {
      model_id: plModel,
      device: plDevice,
      device_index: String(plDeviceIndex),
      timeout: plTimeout,
      session_key: plSession || 'pipeline',
      run_autotest: plChkTest,
      run_autoquant: plChkQuant,
      run_autoeval: plChkEval,
      test_output_dir: plTestOutput,
      autorun_skill: plAutorunSkill,
      scheme: plScheme,
      method: plMethod,
      export_format: plExport,
      quant_output_dir: plQuantOutput,
      autoquant_skill: plAutoquantSkill,
      tasks: plTasks,
      eval_output_dir: plEvalOutput,
      batch_size: plBatchSize,
      max_model_len: plMaxLen,
      gpu_memory_utilization: plGpuMem,
      skill_path: plSkill,
      machine: getMachineConfig(),
    });
  }

  async function runHFDiscover() {
    hfModels = [];
    hfSelectAll = false;
    hfError = '';
    postAndStream('/api/run-hf-discover', {
      days: hfDays,
      limit: hfLimit,
      sort: hfSort,
      pipeline_tag: hfPipelineTag,
      timeout: hfTimeout,
      session_key: hfSession || 'hf_discover',
      machine: getMachineConfig(),
    });
  }

  // Parse model list from session log (agent returns JSON with .models array or raw array)
  function parseModelsFromLog() {
    const logs = tabStates['hfdiscover'].sessionLog;
    for (let i = logs.length - 1; i >= 0; i--) {
      const msg = logs[i].message || '';
      // Try parsing as JSON object with .models key first
      const objMatch = msg.match(/\{[\s\S]*"models"[\s\S]*\}/);
      if (objMatch) {
        try {
          const obj = JSON.parse(objMatch[0]);
          if (Array.isArray(obj.models) && obj.models.length > 0) {
            hfModels = obj.models.map((m: any) => ({ ...m, selected: false }));
            return;
          }
        } catch {}
      }
      // Fallback: look for JSON array directly
      const arrMatch = msg.match(/\[[\s\S]*\]/);
      if (arrMatch) {
        try {
          const arr = JSON.parse(arrMatch[0]);
          if (Array.isArray(arr) && arr.length > 0 && arr[0].id) {
            hfModels = arr.map((m: any) => ({ ...m, selected: false }));
            return;
          }
        } catch {}
      }
    }
  }

  function toggleHFSelectAll() {
    hfSelectAll = !hfSelectAll;
    hfModels = hfModels.map(m => ({ ...m, selected: hfSelectAll }));
  }

  function useSelectedInPipeline() {
    const selected = hfModels.filter(m => m.selected);
    if (selected.length === 0) { alert('Please select at least one model'); return; }
    // Use the first selected model in pipeline form
    plModel = selected[0].id;
    currentTab = 'pipeline';
  }

  function stopCurrent() {
    if (evtSource) {
      evtSource.close();
      evtSource = null;
    }
    isRunning = false;
    appendSessionLog({ timestamp: ts(), type: 'error', message: 'Manually stopped' });
  }

  function downloadSessionLog() {
    const mc = getMachineConfig();
    const url = `${BACKEND}/api/download-session?session_dir=${encodeURIComponent(mc.session_dir)}&container=${encodeURIComponent(mc.container)}`;
    window.open(url, '_blank');
  }

  // ─── Markdown table renderer ─────────────────────────────────────────────
  function escHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderMsg(text: string | undefined | null): string {
    if (!text) return '';
    try {
      const lines = text.split('\n');
      let html = '';
      let tbl: string[] = [];

      const flush = () => {
        if (!tbl.length) return;
        html += '<div class="overflow-x-auto my-1"><table class="border-collapse text-xs w-max">';
        let firstRow = true;
        let hasHeader = tbl.some(l => /^\|[\s\-:|]+\|$/.test(l.trim()));
        for (const l of tbl) {
          const t = l.trim();
          const inner = t.slice(1, -1);
          if (/^[\s\-:|]+$/.test(inner)) { firstRow = false; continue; }
          const cells = inner.split('|');
          const isHdr = firstRow && hasHeader;
          const tag = isHdr ? 'th' : 'td';
          const cls = isHdr
            ? 'border border-slate-400 px-2 py-1 bg-slate-100 font-semibold whitespace-nowrap'
            : 'border border-slate-300 px-2 py-0.5 whitespace-nowrap even:bg-slate-50';
          html += '<tr>' + cells.map(c => `<${tag} class="${cls}">${escHtml(c.trim())}</${tag}>`).join('') + '</tr>';
          if (firstRow) firstRow = false;
        }
        html += '</table></div>';
        tbl = [];
      };

      for (const line of lines) {
        const t = line.trim();
        if (t.startsWith('|') && t.endsWith('|') && t.length > 2) {
          tbl.push(line);
        } else {
          flush();
          html += (t ? `<div>${escHtml(line)}</div>` : '<div class="h-1"></div>');
        }
      }
      flush();
      return html;
    } catch {
      return escHtml(String(text));
    }
  }

  // ─── View xlsx file from container ───────────────────────────────────────
  async function readXlsxFile() {
    if (!xlxFilePath.trim()) { alert('Please enter an Excel file path'); return; }
    xlxFileContent = 'Loading...';
    const mc = getMachineConfig();
    try {
      const res = await fetch(
        `${BACKEND}/api/read-xlsx?path=${encodeURIComponent(xlxFilePath)}&container=${encodeURIComponent(mc.container)}`
      );
      if (!res.ok) { const t = await res.text(); xlxFileContent = 'Error: ' + t; return; }
      const { markdown } = await res.json();
      xlxFileContent = markdown;
    } catch (e: any) {
      xlxFileContent = 'Error: ' + e.message;
    }
  }

  function clearPanels() {
    tabStates[currentTab].sessionLog = [];
    tabStates[currentTab].taskLog = [];
    tabStates[currentTab].collapsedStatus = new Set();
    zimageImgSrcMap[currentTab] = '';
    zimageImgSrcMap = { ...zimageImgSrcMap };
  }

  onDestroy(() => {
    if (evtSource) evtSource.close();
  });

  let sessionLogEl: HTMLElement;
  let taskLogEl: HTMLElement;

  function isNearBottom(el: HTMLElement, threshold = 80): boolean {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
  }

  function onSessionScroll() {
    tabStates[currentTab].sessionUserScrolledUp = !isNearBottom(sessionLogEl);
  }

  function onTaskScroll() {
    tabStates[currentTab].taskUserScrolledUp = !isNearBottom(taskLogEl);
  }

  $: if (sessionLogEl && sessionLog.length > 0) {
    setTimeout(() => {
      if (!tabStates[currentTab].sessionUserScrolledUp) {
        sessionLogEl.scrollTop = sessionLogEl.scrollHeight;
      }
    }, 0);
  }

  $: if (taskLogEl && taskLog.length > 0) {
    setTimeout(() => {
      if (!tabStates[currentTab].taskUserScrolledUp) {
        taskLogEl.scrollTop = taskLogEl.scrollHeight;
      }
    }, 0);
  }

  // Reset scroll position when switching tabs (only if not scrolled up)
  $: if (currentTab && sessionLogEl) {
    setTimeout(() => {
      if (!tabStates[currentTab].sessionUserScrolledUp) {
        sessionLogEl.scrollTop = sessionLogEl.scrollHeight;
      }
    }, 0);
  }
  $: if (currentTab && taskLogEl) {
    setTimeout(() => {
      if (!tabStates[currentTab].taskUserScrolledUp) {
        taskLogEl.scrollTop = taskLogEl.scrollHeight;
      }
    }, 0);
  }

  // ─── Resizable panels ────────────────────────────────────────────────────
  // widths in pixels; null = flex (auto)
  let sessionPanelW: number | null = null;
  let taskPanelW: number | null = null;
  let imgPanelW: number | null = null;
  let panelsContainerEl: HTMLElement;

  type ResizeTarget = 'session-task' | 'task-img';
  let dragging: ResizeTarget | null = null;
  let dragStartX = 0;
  let dragStartSessionW = 0;
  let dragStartTaskW = 0;
  let dragStartImgW = 0;

  function getPanelWidth(el: HTMLElement): number {
    return el.getBoundingClientRect().width;
  }

  function startResize(e: MouseEvent, target: ResizeTarget) {
    if (!panelsContainerEl) return;
    dragging = target;
    dragStartX = e.clientX;
    // snapshot current rendered widths
    const children = panelsContainerEl.children;
    // children order: session | divider | task | [divider | img]
    dragStartSessionW = (children[0] as HTMLElement).getBoundingClientRect().width;
    dragStartTaskW = (children[2] as HTMLElement).getBoundingClientRect().width;
    if (children[4]) dragStartImgW = (children[4] as HTMLElement).getBoundingClientRect().width;
    e.preventDefault();
  }

  function onMouseMove(e: MouseEvent) {
    if (!dragging) return;
    const dx = e.clientX - dragStartX;
    const MIN = 120;
    if (dragging === 'session-task') {
      const newS = Math.max(MIN, dragStartSessionW + dx);
      const newT = Math.max(MIN, dragStartTaskW - dx);
      sessionPanelW = newS;
      taskPanelW = newT;
    } else if (dragging === 'task-img') {
      const newT = Math.max(MIN, dragStartTaskW + dx);
      const newI = Math.max(MIN, dragStartImgW - dx);
      taskPanelW = newT;
      imgPanelW = newI;
    }
  }

  function onMouseUp() {
    dragging = null;
  }
</script>

<div class="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-blue-50"
  role="application"
  aria-label="KernelAgent"
  on:mousemove={onMouseMove}
  on:mouseup={onMouseUp}
  on:mouseleave={onMouseUp}
  style="user-select: {dragging ? 'none' : 'auto'}; cursor: {dragging ? 'col-resize' : 'auto'}"
>
  <!-- Tabs -->
  <div class="flex gap-1 px-4 bg-white/80 backdrop-blur-sm border-b border-blue-200 h-11 items-end shadow-sm">
    <button
      on:click={() => currentTab = 'zimage'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'zimage' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Z-Image
    </button>
    <button
      on:click={() => currentTab = 'excelxlx'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'excelxlx' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Excel-XLX
    </button>
    <button
      on:click={() => currentTab = 'hfdiscover'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'hfdiscover' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      HF Discover
    </button>
    <button
      on:click={() => currentTab = 'pipeline'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'pipeline' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Pipeline
    </button>
  </div>

  <!-- Main Layout -->
  <div class="flex flex-1 overflow-hidden">
    <!-- Sidebar -->
    <div class="w-80 min-w-[280px] max-w-[340px] bg-white/90 backdrop-blur-sm border-r border-blue-200 flex flex-col p-4 gap-3 overflow-y-auto scrollbar-thin shadow-lg">
      <!-- Z-Image Form -->
      {#if currentTab === 'zimage'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Z-Image Agent</h2>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Prompt</label>
          <textarea bind:value={ziPrompt} rows="4" class="w-full bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"></textarea>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Device</label>
          <select bind:value={ziDevice} class="w-full h-9 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
            <option value="xpu">XPU (Intel GPU)</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Steps</label>
            <input type="number" bind:value={ziSteps} min="1" max="50" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Seed</label>
            <input type="number" bind:value={ziSeed} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Output Path</label>
          <input type="text" bind:value={ziOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Session Key</label>
          <input type="text" bind:value={ziSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <button
          on:click={runZImage}
          disabled={isRunning}
          class="h-10 p-2  bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Generate
        </button>
        {#if isRunning}
          <button
            on:click={stopCurrent}
            class="h-10 p-2 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg font-bold text-sm hover:from-red-700 hover:to-red-800 transition-all shadow-md hover:shadow-lg"
          >
            Stop
          </button>
        {/if}
      {/if}

      <!-- Excel-XLX Form -->
      {#if currentTab === 'excelxlx'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Excel-XLX Agent</h2>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Excel File Path (in container)</label>
          <input type="text" bind:value={xlxFilePath} placeholder="/tmp/xxx.xlsx" class="w-full h-8 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm" />
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Instructions</label>
          <textarea bind:value={xlxInstructions} rows="5" placeholder="Read all sheet names and the first 5 rows of each sheet, display in table format" class="w-full bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-3 py-2 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"></textarea>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Skill Path (optional)</label>
          <input type="text" bind:value={xlxSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Session Key</label>
            <input type="text" bind:value={xlxSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Timeout (s)</label>
            <input type="number" bind:value={xlxTimeout} min="60" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <button
          on:click={runExcelXLX}
          disabled={isRunning}
          class="h-10 p-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Run Excel-XLX
        </button>
       
      {/if}

      <!-- Pipeline Form -->
      <!-- HF Discover Form -->
      {#if currentTab === 'hfdiscover'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">HF Model Discover</h2>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs font-semibold text-slate-700 mb-1.5">Days Back</label>
            <input type="number" bind:value={hfDays} min="1" max="365" class="w-full h-8 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm" />
          </div>
          <div class="flex-1">
            <label class="block text-xs font-semibold text-slate-700 mb-1.5">Limit</label>
            <input type="number" bind:value={hfLimit} min="1" max="100" class="w-full h-8 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm" />
          </div>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Sort By</label>
          <select bind:value={hfSort} class="w-full h-9 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
            <option value="createdAt">Created Date</option>
            <option value="downloads">Downloads</option>
            <option value="likes">Likes</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-700 mb-1.5">Pipeline Tag (optional)</label>
          <select bind:value={hfPipelineTag} class="w-full h-9 bg-white border-2 border-slate-300 text-slate-900 rounded-lg px-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
            <option value="">All</option>
            <option value="text-generation">text-generation</option>
            <option value="image-text-to-text">image-text-to-text</option>
            <option value="text-classification">text-classification</option>
            <option value="image-classification">image-classification</option>
            <option value="text2text-generation">text2text-generation</option>
            <option value="automatic-speech-recognition">automatic-speech-recognition</option>
            <option value="image-to-text">image-to-text</option>
          </select>
        </div>
        <button
          on:click={runHFDiscover}
          disabled={isRunning}
          class="h-10 p-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          {isRunning && currentTab === 'hfdiscover' ? '⏳ Agent Running...' : '▶ Discover Models'}
        </button>
        {#if isRunning && currentTab === 'hfdiscover'}
          <button
            on:click={stopCurrent}
            class="h-10 p-2 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg font-bold text-sm hover:from-red-700 hover:to-red-800 transition-all shadow-md hover:shadow-lg"
          >
            Stop
          </button>
        {/if}
        {#if hfError}
          <div class="text-xs text-red-600 font-medium">{hfError}</div>
        {/if}

        <!-- Model List Results -->
        {#if hfModels.length > 0}
          <div class="border-t border-gray-200 pt-3">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-bold text-blue-700">{hfModels.length} Models Found</span>
              <label class="flex items-center gap-1 text-xs text-gray-600 cursor-pointer">
                <input type="checkbox" checked={hfSelectAll} on:change={toggleHFSelectAll} class="w-3.5 h-3.5" />
                All
              </label>
            </div>
            <div class="flex flex-col gap-1 max-h-60 overflow-y-auto scrollbar-thin">
              {#each hfModels as model, i}
                <label class="flex items-start gap-2 p-1.5 rounded hover:bg-blue-50 cursor-pointer text-xs border border-transparent hover:border-blue-200 transition-colors">
                  <input type="checkbox" bind:checked={hfModels[i].selected} class="w-3.5 h-3.5 mt-0.5 flex-none" />
                  <div class="min-w-0">
                    <div class="font-semibold text-slate-800 truncate" title={model.id}>{model.id}</div>
                    <div class="text-[10px] text-slate-500">
                      {model.createdAt?.slice(0, 10) || '?'}
                      {#if model.pipeline_tag} · {model.pipeline_tag}{/if}
                      · ↓{model.downloads} · ♥{model.likes}
                    </div>
                  </div>
                </label>
              {/each}
            </div>
            <button
              on:click={useSelectedInPipeline}
              class="mt-2 h-9 w-full bg-gradient-to-r from-emerald-600 to-emerald-700 text-white rounded-lg font-bold text-sm hover:from-emerald-700 hover:to-emerald-800 transition-all shadow-md hover:shadow-lg"
            >
              → Use in Pipeline
            </button>
          </div>
        {/if}
      {/if}

      {#if currentTab === 'pipeline'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Pipeline</h2>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Model ID</label>
          <input type="text" bind:value={plModel} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Device</label>
          <select bind:value={plDevice} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
            <option value="xpu">XPU (Intel GPU)</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Device Index</label>
            <input type="number" bind:value={plDeviceIndex} min="0" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Timeout/Stage (s)</label>
            <input type="number" bind:value={plTimeout} min="60" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Session Key Prefix</label>
          <input type="text" bind:value={plSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div class="pt-2">
          <label class="block text-xs text-gray-600 mb-2">Stages:</label>
          <label class="flex items-center gap-2 text-sm text-gray-700 mb-1.5 cursor-pointer">
            <input type="checkbox" bind:checked={plChkTest} class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-2 focus:ring-primary-500" />
            Auto-Test
          </label>
          <label class="flex items-center gap-2 text-sm text-gray-700 mb-1.5 cursor-pointer">
            <input type="checkbox" bind:checked={plChkQuant} class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-2 focus:ring-primary-500" />
            Auto-Quant
          </label>
          <label class="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
            <input type="checkbox" bind:checked={plChkEval} class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-2 focus:ring-primary-500" />
            Auto-Eval
          </label>
        </div>

        {#if plChkTest}
          <div class="pt-2 border-t border-gray-200">
            <div class="text-sm font-bold text-blue-700 mb-2 pb-1 border-b border-blue-200">Auto-Test</div>
            <div class="mb-2">
              <label class="block text-xs text-gray-600 mb-1">Test Output Dir</label>
              <input type="text" bind:value={plTestOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            <div>
              <label class="block text-xs text-gray-600 mb-1">Auto-Run Skill</label>
              <input type="text" bind:value={plAutorunSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
          </div>
        {/if}

        {#if plChkQuant || plChkEval}
          <div class="pt-2 border-t border-gray-200">
            <div class="text-sm font-bold text-blue-700 mb-2 pb-1 border-b border-blue-200">{plChkQuant ? 'Auto-Quant' : 'Quantization Config (for eval input derivation)'}</div>
            <div class="flex gap-2 mb-2">
              <div class="flex-1">
                <label class="block text-xs text-gray-600 mb-1">Scheme</label>
                <select bind:value={plScheme} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
                  <option value="W4A16">W4A16</option>
                  <option value="W8A8">W8A8</option>
                  <option value="mxfp4">mxfp4</option>
                  <option value="W4A8">W4A8</option>
                  <option value="W2A16">W2A16</option>
                </select>
              </div>
              {#if plChkQuant}
                <div class="flex-1">
                  <label class="block text-xs text-gray-600 mb-1">Method</label>
                  <select bind:value={plMethod} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
                    <option value="RTN">RTN</option>
                    <option value="GPTQ">GPTQ</option>
                    <option value="AutoRound">AutoRound</option>
                  </select>
                </div>
              {/if}
            </div>
            {#if plChkQuant}
              <div class="mb-2">
                <label class="block text-xs text-gray-600 mb-1">Export Format</label>
                <input type="text" bind:value={plExport} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
              </div>
            {/if}
            <div class="mb-2">
              <label class="block text-xs text-gray-600 mb-1">Quant Output Dir</label>
              <input type="text" bind:value={plQuantOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            {#if plChkQuant}
              <div>
                <label class="block text-xs text-gray-600 mb-1">Auto-Quant Skill</label>
                <input type="text" bind:value={plAutoquantSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
              </div>
            {/if}
            {#if plChkEval}
              <div class="mt-2">
                <label class="block text-xs text-gray-500 mb-1">↳ Eval input (auto-derived)</label>
                <input type="text" value={plEvalInput} readonly class="w-full h-8 bg-gray-100 border border-gray-300 text-gray-600 rounded-md px-2 text-sm cursor-not-allowed" />
              </div>
            {/if}
          </div>
        {/if}

        {#if plChkEval}
          <div class="pt-2 border-t border-gray-200">
            <div class="text-sm font-bold text-blue-700 mb-2 pb-1 border-b border-blue-200">Auto-Eval</div>
            <div class="mb-2">
              <label class="block text-xs text-gray-600 mb-1">Tasks</label>
              <input type="text" bind:value={plTasks} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            <div class="mb-2">
              <label class="block text-xs text-gray-600 mb-1">Eval Output Dir</label>
              <input type="text" bind:value={plEvalOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            <div class="flex gap-2 mb-2">
              <div class="flex-1">
                <label class="block text-xs text-gray-600 mb-1">Batch Size</label>
                <input type="number" bind:value={plBatchSize} min="1" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
              </div>
              <div class="flex-1">
                <label class="block text-xs text-gray-600 mb-1">Max Model Len</label>
                <input type="number" bind:value={plMaxLen} min="128" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
              </div>
            </div>
            <div class="mb-2">
              <label class="block text-xs text-gray-600 mb-1">GPU Mem Util</label>
              <input type="number" bind:value={plGpuMem} min="0.1" max="1.0" step="0.05" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            <div>
              <label class="block text-xs text-gray-600 mb-1">Skill Path</label>
              <input type="text" bind:value={plSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
          </div>
        {/if}

        <button
          on:click={runPipeline}
          disabled={isRunning}
          class="h-10 p-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Run Pipeline
        </button>
        {#if isRunning}
          <button
            on:click={stopCurrent}
            class="h-10 p-2 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg font-bold text-sm hover:from-red-700 hover:to-red-800 transition-all shadow-md hover:shadow-lg"
          >
            Stop
          </button>
        {/if}
      {/if}

      <!-- Machine Config -->
      <details bind:open={showMachineConfig} class="border-t border-gray-200 pt-3">
        <summary class="cursor-pointer text-sm text-blue-700 py-1.5 font-bold hover:text-blue-800 transition-colors">⚙ Machine Config</summary>
        <div class="flex flex-col gap-2 mt-2">
          <div class="text-xs font-bold text-slate-600 tracking-wide uppercase">SSH</div>
          <div class="flex gap-2">
            <div class="flex-1">
              <label class="block text-xs text-gray-600 mb-1">Host</label>
              <input type="text" bind:value={mcHost} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
            <div class="flex-1">
              <label class="block text-xs text-gray-600 mb-1">User</label>
              <input type="text" bind:value={mcUser} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
            </div>
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Password</label>
            <input type="password" bind:value={mcPassword} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="text-xs font-bold text-slate-600 tracking-wide uppercase mt-1">Docker & Paths</div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Container</label>
            <input type="text" bind:value={mcContainer} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Repo Root (in container)</label>
            <input type="text" bind:value={mcRepoRoot} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Workdir (in container)</label>
            <input type="text" bind:value={mcWorkdir} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Output Root (in container)</label>
            <input type="text" bind:value={mcOutputRoot} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Session Dir (in container)</label>
            <input type="text" bind:value={mcSessionDir} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">MiniMax API Key</label>
            <input type="text" bind:value={mcMinimaxKey} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
      </details>

      <button
        on:click={clearPanels}
        class="h-10 p-2 bg-gradient-to-r from-slate-200 to-slate-300 text-slate-700 rounded-lg font-bold text-sm hover:from-slate-300 hover:to-slate-400 transition-all shadow-md hover:shadow-lg"
      >
         Clear Panels
      </button>
    </div>

    <!-- Main Panels -->
    <div class="flex flex-1 overflow-hidden" bind:this={panelsContainerEl}>
      <!-- Session Panel -->
      <div class="flex flex-col overflow-hidden bg-white shadow-inner"
        style={sessionPanelW !== null ? `width:${sessionPanelW}px; flex:none` : 'flex:1'}>
        <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-blue-50 to-slate-50 flex items-center justify-between">
          <span>Agent Session</span>
          <div class="flex items-center gap-2">
            {#if !isRunning && tabStates[currentTab].sessionLog.length > 0}
              <button
                on:click={downloadSessionLog}
                class="text-[10px] px-2 py-0.5 rounded bg-emerald-500 text-white font-semibold hover:bg-emerald-600 transition-colors"
                title="Download raw session JSONL from container"
              >⬇ Session</button>
            {/if}
          {#if tabStates[currentTab].sessionUserScrolledUp}
            <button
              on:click={() => { tabStates[currentTab].sessionUserScrolledUp = false; sessionLogEl.scrollTop = sessionLogEl.scrollHeight; }}
              class="text-[10px] px-2 py-0.5 rounded bg-blue-500 text-white font-semibold hover:bg-blue-600 transition-colors"
              title="Scroll to bottom"
            >↓ bottom</button>
          {/if}
          </div>
        </div>
        <div bind:this={sessionLogEl} on:scroll={onSessionScroll} class="flex-1 overflow-y-auto p-4 text-xs font-mono leading-relaxed scrollbar-thin bg-slate-50/50">
          {#each sessionLog as item, idx}
            <div class="py-1 hover:bg-white/60 px-2 -mx-2 rounded transition-colors break-words">
              <span class="text-slate-400 text-[10px] select-none font-semibold mr-1.5">{item.timestamp}</span><!--
              -->{#if item.type === 'status'}
                <button
                  on:click={() => toggleCollapsedStatus(idx)}
                  class="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 font-bold mr-1 hover:bg-blue-200 cursor-pointer select-none"
                  title="Click to expand/collapse"
                >{tabStates[currentTab].collapsedStatus.has(idx) ? '▷' : '▶'}</button><span class="text-blue-700 font-semibold">{#if tabStates[currentTab].collapsedStatus.has(idx)}{item.message.slice(0, 60)}{item.message.length > 60 ? '…' : ''}{:else}{item.message}{/if}</span>
              {:else if item.type === 'done'}
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-bold mr-1">✓</span><span class="text-green-700 font-bold">{@html renderMsg(item.message)}</span>
              {:else if item.type === 'error'}
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-700 font-bold mr-1">✗</span><span class="text-red-700 font-bold">{item.message}</span>
              {:else if item.type === 'thinking'}
                <button
                  on:click={() => toggleCollapsedStatus(idx)}
                  class="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 text-slate-700 font-semibold mr-1 hover:bg-slate-300 cursor-pointer"
                  title="Click to expand/collapse"
                >{tabStates[currentTab].collapsedStatus.has(idx) ? '[Think ▷]' : '[Think ▼]'}</button>{#if !tabStates[currentTab].collapsedStatus.has(idx)}<span class="text-slate-600 italic">{item.message}</span>{/if}
              {:else if item.type === 'tool_call'}
                <button
                  on:click={() => toggleCollapsedStatus(idx)}
                  class="text-[10px] px-1.5 py-0.5 rounded bg-purple-200 text-purple-900 font-semibold mr-1 hover:bg-purple-300 cursor-pointer"
                  title="Click to expand/collapse"
                >{tabStates[currentTab].collapsedStatus.has(idx) ? '[Tool ▷]' : '[Tool ▼]'}</button>{#if !tabStates[currentTab].collapsedStatus.has(idx)}<span class="text-purple-700 font-medium">{item.message}</span>{/if}
              {:else if item.type === 'tool_result'}
                <button
                  on:click={() => toggleCollapsedStatus(idx)}
                  class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-200 text-emerald-900 font-semibold mr-1 hover:bg-emerald-300 cursor-pointer"
                  title="Click to expand/collapse"
                >{tabStates[currentTab].collapsedStatus.has(idx) ? '[Result ▷]' : '[Result ▼]'}</button>{#if !tabStates[currentTab].collapsedStatus.has(idx)}<span class="text-emerald-700 font-medium">{@html renderMsg(item.message)}</span>{/if}
              {:else}
                {#if item.tag}<span class="text-[10px] px-1.5 py-0.5 rounded bg-blue-200 text-blue-900 font-semibold mr-1">[{item.tag}]</span>{/if}<span class="text-slate-800">{@html renderMsg(item.message)}</span>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Drag handle: Session ↔ Task -->
      <div
        role="separator"
        aria-label="Resize panels"
        class="w-1.5 flex-none bg-blue-100 hover:bg-blue-400 active:bg-blue-500 cursor-col-resize transition-colors border-x border-blue-200 flex items-center justify-center group"
        on:mousedown={(e) => startResize(e, 'session-task')}
        title="Drag to resize"
      >
        <div class="w-px h-8 bg-blue-300 group-hover:bg-blue-500 rounded transition-colors"></div>
      </div>

      <!-- Task Log Panel -->
      <div class="flex flex-col overflow-hidden bg-white shadow-inner"
        style={taskPanelW !== null ? `width:${taskPanelW}px; flex:none` : 'flex:1'}>
        <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-slate-50 to-blue-50 flex items-center justify-between">
          <span>Task Log</span>
          {#if tabStates[currentTab].taskUserScrolledUp}
            <button
              on:click={() => { tabStates[currentTab].taskUserScrolledUp = false; taskLogEl.scrollTop = taskLogEl.scrollHeight; }}
              class="text-[10px] px-2 py-0.5 rounded bg-blue-500 text-white font-semibold hover:bg-blue-600 transition-colors"
              title="Scroll to bottom"
            >↓ bottom</button>
          {/if}
        </div>
        <div bind:this={taskLogEl} on:scroll={onTaskScroll} class="flex-1 overflow-y-auto p-4 text-xs font-mono leading-relaxed scrollbar-thin bg-slate-50/50">
          {#each taskLog as item}
            <div class="py-1 break-words hover:bg-white/60 px-2 -mx-2 rounded transition-colors">
              <span class="text-slate-500 text-[10px] mr-2 select-none font-semibold">{item.timestamp}</span>
              <span class="
                {item.type === 'zimage_log' ? 'text-slate-600' : ''}
                {item.type === 'log' ? 'text-slate-800' : ''}
              ">{item.message}</span>
            </div>
          {/each}
        </div>
      </div>

      {#if (currentTab === 'zimage' && zimageImgSrc) || (currentTab === 'excelxlx' && xlxFileContent)}
        <!-- Drag handle: Task ↔ Image/Preview -->
        <div
          role="separator"
          aria-label="Resize panels"
          class="w-1.5 flex-none bg-blue-100 hover:bg-blue-400 active:bg-blue-500 cursor-col-resize transition-colors border-x border-blue-200 flex items-center justify-center group"
          on:mousedown={(e) => startResize(e, 'task-img')}
          title="Drag to resize"
        >
          <div class="w-px h-8 bg-blue-300 group-hover:bg-blue-500 rounded transition-colors"></div>
        </div>
      {/if}

      {#if currentTab === 'zimage' && zimageImgSrc}
        <div class="flex flex-col overflow-hidden bg-white"
          style={imgPanelW !== null ? `width:${imgPanelW}px; flex:none` : 'flex:1'}>
          <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-blue-50 to-slate-50">
            Generated Image
          </div>
          <div class="flex-1 flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 to-blue-50 overflow-hidden">
            <img src={zimageImgSrc} alt="" class="max-w-full max-h-full object-contain rounded-xl shadow-2xl border-2 border-blue-200" />
          </div>
        </div>
      {/if}

      {#if currentTab === 'excelxlx' && xlxFileContent}
        <div class="flex flex-col overflow-hidden bg-white border-l border-blue-200"
          style={imgPanelW !== null ? `width:${imgPanelW}px; flex:none` : 'flex:1'}>
          <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-emerald-50 to-slate-50 flex items-center justify-between">
            <span>File Preview: {xlxFilePath}</span>
            <button on:click={() => xlxFileContent = ''} class="text-slate-400 hover:text-slate-700 ml-2 font-bold">✕</button>
          </div>
          <div class="flex-1 overflow-auto p-4 text-xs font-mono bg-slate-50/50">
            {@html renderMsg(xlxFileContent)}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
