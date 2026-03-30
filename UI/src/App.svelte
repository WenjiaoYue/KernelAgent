<script lang="ts">
  import { onDestroy } from 'svelte';

  type Tab = 'zimage' | 'autotest' | 'autoquant' | 'autoeval' | 'pipeline';
  type LogItem = {
    timestamp: string;
    type: string;
    tag?: string;
    message: string;
  };

  let currentTab: Tab = 'zimage';
  let sessionLog: LogItem[] = [];
  let taskLog: LogItem[] = [];
  let evtSource: EventSource | null = null;
  let isRunning = false;
  let showMachineConfig = false;
  let zimageImgSrc = '';

  const BACKEND = window.location.origin;

  // Z-Image form
  let ziPrompt = 'a cup of coffee on the wooden table, photorealistic, 8k';
  let ziDevice = 'xpu';
  let ziSteps = 9;
  let ziSeed = 42;
  let ziOutput = '/storage/lkk/inference/zimage/zimage_output.png';
  let ziSession = 'zimage_task';

  // Auto-Test form
  let atModel = 'Qwen/Qwen3-0.6B';
  let atDevice = 'xpu';
  let atDeviceIndex = 0;
  let atTimeout = 7200;
  let atOutput = '/storage/lkk/inference';
  let atSkill = '/root/.openclaw/workspace/skills/auto_run/SKILL.md';
  let atSession = 'autotest_task';

  // Auto-Quant form
  let aqModel = 'Qwen/Qwen3-0.6B';
  let aqScheme = 'W4A16';
  let aqMethod = 'RTN';
  let aqExport = 'auto_round:auto_gptq';
  let aqDevice = 'xpu';
  let aqDeviceIndex = 0;
  let aqTimeout = 7200;
  let aqOutput = '/kaokao/quantized';
  let aqSkill = '/root/.openclaw/workspace/skills/auto_quant/SKILL.md';
  let aqSession = 'autoquant_task';

  // Auto-Eval form
  let aeModelPath = '/kaokao/quantized/Qwen_Qwen3-0.6B-W4A16';
  let aeTasks = 'piqa';
  let aeOutput = '/kaokao/lm_eval_results/Qwen_Qwen3-0.6B';
  let aeDevice = 'xpu';
  let aeDeviceIndex = 0;
  let aeBatchSize = 8;
  let aeMaxLen = 8192;
  let aeGpuMem = 0.8;
  let aeSkill = '/root/.openclaw/workspace/skills/auto_eval/SKILL.md';
  let aeSession = 'autoeval_task';
  let aeTimeout = 7200;

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

  function appendLog(log: LogItem[], item: LogItem): LogItem[] {
    return [...log, item];
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
      taskLog = appendLog(taskLog, { timestamp: ts(), type: 'zimage_log', message: msg });
      return;
    }

    if (type === 'transcript') {
      const tagMap: Record<string, string> = { thinking: 'Think', tool_call: 'Tool', tool_result: 'Result', text: 'Text' };
      sessionLog = appendLog(sessionLog, {
        timestamp: ts(),
        type: sub || 'text',
        tag: tagMap[sub] || sub,
        message: msg
      });
      return;
    }

    if (type === 'status') {
      sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'status', message: '▶ ' + msg });
      return;
    }

    if (type === 'log') {
      taskLog = appendLog(taskLog, { timestamp: ts(), type: 'log', message: msg });
      return;
    }

    if (type === 'done') {
      sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'done', message: '✓ ' + msg });
      if (currentTab === 'zimage' && obj.meta && obj.meta.output) {
        const mc = getMachineConfig();
        zimageImgSrc = `${BACKEND}/api/image?path=${encodeURIComponent(obj.meta.output)}&container=${encodeURIComponent(mc.container)}`;
      }
      return;
    }

    if (type === 'error') {
      sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'error', message: '✗ ' + msg });
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
    sessionLog = [];
    taskLog = [];
    sessionUserScrolledUp = false;
    taskUserScrolledUp = false;
    isRunning = true;

    let res: Response;
    try {
      res = await fetch(`${BACKEND}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err: any) {
      sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'error', message: 'Connection failed: ' + err.message });
      isRunning = false;
      return;
    }

    if (!res.ok) {
      const t = await res.text();
      sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'error', message: `HTTP ${res.status}: ${t}` });
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

  function runAutoTest() {
    if (!atModel.trim()) {
      alert('Please enter a Model ID');
      return;
    }
    postAndStream('/api/run-autotest', {
      model_id: atModel,
      device: atDevice,
      device_index: String(atDeviceIndex),
      output_dir: atOutput,
      skill_path: atSkill,
      session_key: atSession || 'autotest_task',
      timeout: atTimeout,
      machine: getMachineConfig(),
    });
  }

  function runAutoQuant() {
    if (!aqModel.trim()) {
      alert('Please enter a Model ID');
      return;
    }
    postAndStream('/api/run-autoquant', {
      model_id: aqModel,
      scheme: aqScheme,
      method: aqMethod,
      export_format: aqExport,
      device: aqDevice,
      device_index: String(aqDeviceIndex),
      output_dir: aqOutput,
      skill_path: aqSkill,
      session_key: aqSession || 'autoquant_task',
      timeout: aqTimeout,
      machine: getMachineConfig(),
    });
  }

  function runAutoEval() {
    if (!aeModelPath.trim()) {
      alert('Please enter a Model Path');
      return;
    }
    postAndStream('/api/run-autoeval', {
      model_path: aeModelPath,
      tasks: aeTasks,
      output_dir: aeOutput,
      device: aeDevice,
      device_index: String(aeDeviceIndex),
      batch_size: aeBatchSize,
      max_model_len: aeMaxLen,
      gpu_memory_utilization: aeGpuMem,
      skill_path: aeSkill,
      session_key: aeSession || 'autoeval_task',
      timeout: aeTimeout,
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

  function stopCurrent() {
    if (evtSource) {
      evtSource.close();
      evtSource = null;
    }
    isRunning = false;
    sessionLog = appendLog(sessionLog, { timestamp: ts(), type: 'error', message: 'Manually stopped' });
  }

  function clearPanels() {
    sessionLog = [];
    taskLog = [];
  }

  onDestroy(() => {
    if (evtSource) evtSource.close();
  });

  let sessionLogEl: HTMLElement;
  let taskLogEl: HTMLElement;
  let sessionUserScrolledUp = false;
  let taskUserScrolledUp = false;

  function isNearBottom(el: HTMLElement, threshold = 80): boolean {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
  }

  function onSessionScroll() {
    sessionUserScrolledUp = !isNearBottom(sessionLogEl);
  }

  function onTaskScroll() {
    taskUserScrolledUp = !isNearBottom(taskLogEl);
  }

  $: if (sessionLogEl && sessionLog.length > 0) {
    setTimeout(() => {
      if (!sessionUserScrolledUp) {
        sessionLogEl.scrollTop = sessionLogEl.scrollHeight;
      }
    }, 0);
  }

  $: if (taskLogEl && taskLog.length > 0) {
    setTimeout(() => {
      if (!taskUserScrolledUp) {
        taskLogEl.scrollTop = taskLogEl.scrollHeight;
      }
    }, 0);
  }
</script>

<div class="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-blue-50">
  <!-- Tabs -->
  <div class="flex gap-1 px-4 bg-white/80 backdrop-blur-sm border-b border-blue-200 h-11 items-end shadow-sm">
    <button
      on:click={() => currentTab = 'zimage'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'zimage' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Z-Image
    </button>
    <button
      on:click={() => currentTab = 'autotest'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'autotest' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Auto-Test
    </button>
    <button
      on:click={() => currentTab = 'autoquant'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'autoquant' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Auto-Quant
    </button>
    <button
      on:click={() => currentTab = 'autoeval'}
      class="px-6 py-2 text-sm font-semibold border border-b-0 rounded-t-lg transition-all {currentTab === 'autoeval' ? 'bg-white text-blue-700 border-blue-200 shadow-sm' : 'bg-slate-100/80 text-slate-600 border-transparent hover:bg-slate-200/80 hover:text-slate-900'}"
    >
      Auto-Eval
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

      <!-- Auto-Test Form -->
      {#if currentTab === 'autotest'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Auto-Test</h2>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Model ID</label>
          <input type="text" bind:value={atModel} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Device</label>
          <select bind:value={atDevice} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="xpu">XPU (Intel GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Device Index</label>
            <input type="number" bind:value={atDeviceIndex} min="0" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Timeout (s)</label>
            <input type="number" bind:value={atTimeout} min="60" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Output Dir</label>
          <input type="text" bind:value={atOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Skill Path</label>
          <input type="text" bind:value={atSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Session Key</label>
          <input type="text" bind:value={atSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <button
          on:click={runAutoTest}
          disabled={isRunning}
          class="h-10 p-2  bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Run Test
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

      <!-- Auto-Quant Form -->
      {#if currentTab === 'autoquant'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Auto-Quant</h2>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Model ID</label>
          <input type="text" bind:value={aqModel} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Scheme</label>
            <select bind:value={aqScheme} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
              <option value="W4A16">W4A16</option>
              <option value="W8A8">W8A8</option>
              <option value="mxfp4">mxfp4</option>
              <option value="W4A8">W4A8</option>
              <option value="W2A16">W2A16</option>
            </select>
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Method</label>
            <select bind:value={aqMethod} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
              <option value="RTN">RTN</option>
              <option value="GPTQ">GPTQ</option>
              <option value="AutoRound">AutoRound</option>
            </select>
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Export Format</label>
          <input type="text" bind:value={aqExport} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Device</label>
          <select bind:value={aqDevice} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
            <option value="xpu">XPU (Intel GPU)</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Device Index</label>
            <input type="number" bind:value={aqDeviceIndex} min="0" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Timeout (s)</label>
            <input type="number" bind:value={aqTimeout} min="60" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Output Dir</label>
          <input type="text" bind:value={aqOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Skill Path</label>
          <input type="text" bind:value={aqSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Session Key</label>
          <input type="text" bind:value={aqSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <button
          on:click={runAutoQuant}
          disabled={isRunning}
          class="h-10 p-2  bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Quantize
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

      <!-- Auto-Eval Form -->
      {#if currentTab === 'autoeval'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Auto-Eval</h2>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Model Path</label>
          <input type="text" bind:value={aeModelPath} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Tasks</label>
          <input type="text" bind:value={aeTasks} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Output Dir</label>
          <input type="text" bind:value={aeOutput} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Device</label>
          <select bind:value={aeDevice} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="xpu">XPU (Intel GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Device Index</label>
            <input type="number" bind:value={aeDeviceIndex} min="0" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Batch Size</label>
            <input type="number" bind:value={aeBatchSize} min="1" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div class="flex gap-2">
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">Max Model Len</label>
            <input type="number" bind:value={aeMaxLen} min="128" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
          <div class="flex-1">
            <label class="block text-xs text-gray-600 mb-1">GPU Mem Util</label>
            <input type="number" bind:value={aeGpuMem} min="0.1" max="1.0" step="0.05" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Skill Path</label>
          <input type="text" bind:value={aeSkill} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Session Key</label>
          <input type="text" bind:value={aeSession} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Timeout (s)</label>
          <input type="number" bind:value={aeTimeout} min="60" class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <button
          on:click={runAutoEval}
          disabled={isRunning}
          class="h-10 p-2  bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
        >
          ▶ Evaluate
        </button>
        {#if isRunning}
          <button
            on:click={stopCurrent}
            class="h-10 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg font-bold text-sm hover:from-red-700 hover:to-red-800 transition-all shadow-md hover:shadow-lg"
          >
           Stop
          </button>
        {/if}
      {/if}

      <!-- Pipeline Form -->
      {#if currentTab === 'pipeline'}
        <h2 class="text-base font-bold text-blue-700 tracking-wide mb-1">Pipeline</h2>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Model ID</label>
          <input type="text" bind:value={plModel} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" />
        </div>
        <div>
          <label class="block text-xs text-gray-600 mb-1">Device</label>
          <select bind:value={plDevice} class="w-full h-8 bg-white border border-gray-300 text-gray-900 rounded-md px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="xpu">XPU (Intel GPU)</option>
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
          class="h-10 p-2  bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg font-bold text-sm hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none"
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
    <div class="flex flex-1 overflow-hidden">
      <div class="flex-1 flex flex-col border-r border-blue-200 overflow-hidden bg-white shadow-inner">
        <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-blue-50 to-slate-50">
          Agent Session
        </div>
        <div bind:this={sessionLogEl} on:scroll={onSessionScroll} class="flex-1 overflow-y-auto p-4 text-xs font-mono leading-relaxed scrollbar-thin bg-slate-50/50">
          {#each sessionLog as item}
            <div class="py-1 break-words hover:bg-white/60 px-2 -mx-2 rounded transition-colors">
              <span class="text-slate-500 text-[10px] mr-2 select-none font-semibold">{item.timestamp}</span>
              {#if item.tag}
                <span class="text-[10px] mr-1.5 px-2 py-0.5 rounded-md align-middle font-semibold
                  {item.type === 'thinking' ? 'bg-slate-200 text-slate-700' : ''}
                  {item.type === 'tool_call' ? 'bg-purple-200 text-purple-900' : ''}
                  {item.type === 'tool_result' ? 'bg-emerald-200 text-emerald-900' : ''}
                  {item.type === 'text' ? 'bg-blue-200 text-blue-900' : ''}
                ">[{item.tag}]</span>
              {/if}
              <span class="
                {item.type === 'thinking' ? 'text-slate-600 italic' : ''}
                {item.type === 'tool_call' ? 'text-purple-700 font-medium' : ''}
                {item.type === 'tool_result' ? 'text-emerald-700 font-medium' : ''}
                {item.type === 'text' ? 'text-slate-800' : ''}
                {item.type === 'status' ? 'text-blue-700 font-semibold' : ''}
                {item.type === 'done' ? 'text-green-700 font-bold' : ''}
                {item.type === 'error' ? 'text-red-700 font-bold' : ''}
              ">{item.message}</span>
            </div>
          {/each}
        </div>
      </div>

      <div class="flex-1 flex flex-col {currentTab === 'zimage' && zimageImgSrc ? 'border-r border-blue-200' : ''} overflow-hidden bg-white shadow-inner">
        <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-slate-50 to-blue-50">
          Task Log
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

      {#if currentTab === 'zimage' && zimageImgSrc}
        <div class="flex-1 flex flex-col overflow-hidden bg-white">
          <div class="px-4 py-3 text-xs font-bold tracking-wider uppercase text-blue-800 border-b-2 border-blue-200 bg-gradient-to-r from-blue-50 to-slate-50">
            Generated Image
          </div>
          <div class="flex-1 flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 to-blue-50 overflow-hidden">
            <img src={zimageImgSrc} alt="" class="max-w-full max-h-full object-contain rounded-xl shadow-2xl border-2 border-blue-200" />
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
