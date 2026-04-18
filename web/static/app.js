// ============================================================
// 数据层
// ============================================================

const PIPE_REGISTRY = {
    'unify-background': {
        type: 'unify-background',
        label: '背景统一',
        endpoint: '/unify-background',
        params: [
            { key: 'tolerance', label: '容差', type: 'range',
              min: 1, max: 20, step: 0.5, default: 5.0 },
            { key: 'target_color', label: '目标色', type: 'color',
              default: '#808080' }
        ],
        resultType: 'default'
    },
    'pixelize': {
        type: 'pixelize',
        label: '像素化',
        endpoint: '/pixelize',
        params: [
            { key: 'cell_size', label: '像素化程度', type: 'select',
              options: [
                { value: 2, label: '2 (轻微)' },
                { value: 3, label: '3' },
                { value: 4, label: '4 (中等)' },
                { value: 5, label: '5' },
                { value: 6, label: '6' },
                { value: 7, label: '7' },
                { value: 8, label: '8 (强烈)' }
              ],
              default: 4 }
        ],
        resultType: 'pixelated'
    },
    'optimize-colors': {
        type: 'optimize-colors',
        label: '颜色优化',
        endpoint: '/optimize-colors',
        params: [
            { key: 'target_colors', label: '目标颜色数', type: 'number',
              min: 2, max: 256, default: 32 }
        ],
        resultType: 'pixelated'
    },
    'grayscale': {
        type: 'grayscale',
        label: '灰度',
        endpoint: '/grayscale',
        params: [
            { key: 'gray_levels', label: '灰阶级数', type: 'select',
              options: [
                { value: 0, label: '连续灰度' },
                { value: 2, label: '2 级' },
                { value: 4, label: '4 级 (GameBoy)' },
                { value: 8, label: '8 级 (阴影)' },
                { value: 16, label: '16 级' },
                { value: 32, label: '32 级' }
              ],
              default: 4 }
        ],
        resultType: 'pixelated'
    },
    'palette-map': {
        type: 'palette-map',
        label: '调色板映射',
        endpoint: '/palette-map',
        params: [
            { key: 'palette', label: '调色板', type: 'select',
              options: [
                { value: 'pico-8', label: 'PICO-8 (16色)' },
                { value: 'gameboy', label: 'GameBoy (4色)' },
                { value: 'nes', label: 'NES (54色)' },
                { value: 'c64', label: 'Commodore 64 (16色)' },
                { value: 'endesga-32', label: 'Endesga 32 (32色)' }
              ],
              default: 'pico-8' },
            { key: 'dither', label: '抖动', type: 'select',
              options: [
                { value: 'none', label: '无' },
                { value: 'ordered', label: '有序' },
                { value: 'floyd-steinberg', label: 'Floyd-Steinberg' }
              ],
              default: 'none' }
        ],
        resultType: 'pixelated'
    }
};

const PIPELINE_PRESETS = {
    'default': { label: '默认（像素化 → 颜色优化）', pipes: ['pixelize', 'optimize-colors'] },
    'with-bg': { label: '含背景统一', pipes: ['unify-background', 'pixelize', 'optimize-colors'] },
    'bg-only': { label: '仅背景统一', pipes: ['unify-background'] },
    'gameboy': { label: 'GameBoy 风格', pipes: ['pixelize', 'grayscale', 'palette-map'] }
};

let _pipeIdCounter = 0;
function nextPipeId() { return 'pipe_' + (_pipeIdCounter++); }

function defaultParams(pipeType) {
    const params = {};
    for (const p of PIPE_REGISTRY[pipeType].params) params[p.key] = p.default;
    return params;
}

function createPipelineState(pipeTypes) {
    const pipes = pipeTypes.map(type => ({ type, id: nextPipeId(), params: defaultParams(type) }));
    const blobs = {};
    for (const p of pipes) blobs[p.id] = null;
    return { pipes, blobs, executedUpTo: -1 };
}

function addPipe(state, type, index) {
    const pipe = { type, id: nextPipeId(), params: defaultParams(type) };
    if (index == null) index = state.pipes.length;
    state.pipes.splice(index, 0, pipe);
    state.blobs[pipe.id] = null;
    resetFrom(state, index);
    return pipe;
}

function removePipe(state, id) {
    const idx = state.pipes.findIndex(p => p.id === id);
    if (idx === -1) return;
    state.pipes.splice(idx, 1);
    delete state.blobs[id];
    resetFrom(state, idx);
}

function movePipe(state, fromIndex, toIndex) {
    const adjusted = fromIndex < toIndex ? toIndex - 1 : toIndex;
    if (adjusted === fromIndex) return;
    const [pipe] = state.pipes.splice(fromIndex, 1);
    state.pipes.splice(adjusted, 0, pipe);
    resetFrom(state, Math.min(fromIndex, adjusted));
}

function updateParam(state, id, key, value) {
    const pipe = state.pipes.find(p => p.id === id);
    if (!pipe) return;
    pipe.params[key] = value;
    resetFrom(state, state.pipes.indexOf(pipe));
}

function getInputBlob(state, pipeIndex) {
    if (pipeIndex === 0) return null;
    return state.blobs[state.pipes[pipeIndex - 1].id];
}

function resetFrom(state, pipeIndex) {
    for (let i = pipeIndex; i < state.pipes.length; i++) state.blobs[state.pipes[i].id] = null;
    if (state.executedUpTo >= pipeIndex) state.executedUpTo = pipeIndex - 1;
}

// --- Batch 数据模型 ---
// Batch 配置格式（Web/Python 共享）:
// { name: string, tile_size: number, configs: [{ name: string, pipeline: [{ type, params }] }] }

function createBatch(name, tileSize) {
    return { name: name || '未命名 Batch', tile_size: tileSize || 16, configs: [] };
}

function addBatchConfig(batch, name, pipelineSteps) {
    const config = {
        name: name || ('配置 ' + (batch.configs.length + 1)),
        pipeline: pipelineSteps || []
    };
    batch.configs.push(config);
    return config;
}

function removeBatchConfig(batch, index) {
    if (index >= 0 && index < batch.configs.length) batch.configs.splice(index, 1);
}

function moveBatchConfig(batch, from, to) {
    if (from === to || from < 0 || from >= batch.configs.length) return;
    const [item] = batch.configs.splice(from, 1);
    const adj = from < to ? to - 1 : to;
    batch.configs.splice(adj, 0, item);
}

function updateBatchConfig(batch, index, updates) {
    if (index < 0 || index >= batch.configs.length) return;
    const cfg = batch.configs[index];
    if (updates.name !== undefined) cfg.name = updates.name;
    if (updates.pipeline !== undefined) cfg.pipeline = updates.pipeline;
}

// 从管线配置生成 pipeline 步骤数组（用于 batch config）
function pipelineToSteps(state) {
    return state.pipes.map(p => ({ type: p.type, params: { ...p.params } }));
}

// 从 pipeline 步骤数组验证（确保所有 type 有效）
function validatePipelineSteps(steps) {
    if (!Array.isArray(steps)) return [];
    return steps.filter(s => s.type && PIPE_REGISTRY[s.type]).map(s => ({
        type: s.type,
        params: { ...defaultParams(s.type), ...(s.params || {}) }
    }));
}

// localStorage
const STORAGE_KEY = 'pixelization_pipeline';
const BATCHES_KEY = 'pixelization_batches';

function serializePipeline(state) {
    return JSON.stringify({ pipes: state.pipes.map(p => ({ type: p.type, params: p.params })) });
}

function deserializePipeline(json) {
    const data = JSON.parse(json);
    const pipes = data.pipes.filter(p => PIPE_REGISTRY[p.type]).map(p => ({
        type: p.type, id: nextPipeId(), params: { ...defaultParams(p.type), ...p.params }
    }));
    if (pipes.length === 0) return createPipelineState(PIPELINE_PRESETS['default'].pipes);
    const blobs = {};
    for (const p of pipes) blobs[p.id] = null;
    return { pipes, blobs, executedUpTo: -1 };
}

function savePipeline(state) {
    try { localStorage.setItem(STORAGE_KEY, serializePipeline(state)); } catch (e) {}
}

function loadPipeline() {
    try {
        const json = localStorage.getItem(STORAGE_KEY);
        if (json) return deserializePipeline(json);
    } catch (e) {}
    return createPipelineState(PIPELINE_PRESETS['default'].pipes);
}

// Batch 持久化
function saveBatches(batches) {
    try { localStorage.setItem(BATCHES_KEY, JSON.stringify(batches)); } catch (e) {}
}

function loadBatches() {
    try {
        const json = localStorage.getItem(BATCHES_KEY);
        if (json) {
            const data = JSON.parse(json);
            if (Array.isArray(data)) return data;
        }
    } catch (e) {}
    return [];
}

function validateBatchJSON(data) {
    if (!data || typeof data !== 'object') return null;
    if (!data.name || typeof data.name !== 'string') return null;
    if (typeof data.tile_size !== 'number' || data.tile_size < 1) return null;
    if (!Array.isArray(data.configs)) return null;
    for (const cfg of data.configs) {
        if (!cfg.name || typeof cfg.name !== 'string') return null;
        if (!Array.isArray(cfg.pipeline)) return null;
        for (const step of cfg.pipeline) {
            if (!step.type || !PIPE_REGISTRY[step.type]) return null;
        }
    }
    return data;
}

// ============================================================
// 状态 & DOM 引用
// ============================================================

let pipelineState = loadPipeline();
let selectedFile = null;
let isExecuting = false;
let pipelineName = '默认管线';
const zoomScales = {};

const $ = id => document.getElementById(id);
const uploadArea = $('uploadArea');
const fileInput = $('fileInput');
const previewImg = $('previewImg');
const pipelineFlow = $('pipelineFlow');
const errorEl = $('error');
const loadingEl = $('loading');
const loadingText = $('loadingText');
const executeAllBtn = $('executeAllBtn');

// ============================================================
// 工具函数
// ============================================================

function showError(msg) { errorEl.textContent = msg; errorEl.classList.add('show'); }
function hideError() { errorEl.classList.remove('show'); }
function showLoading(text) { loadingText.textContent = text || '处理中...'; loadingEl.classList.add('show'); }
function hideLoading() { loadingEl.classList.remove('show'); }

function countColors(imgEl) {
    const c = document.createElement('canvas');
    c.width = imgEl.naturalWidth; c.height = imgEl.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(imgEl, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    const s = new Set();
    for (let i = 0; i < d.length; i += 4) s.add((d[i] << 16) | (d[i+1] << 8) | d[i+2]);
    return s.size;
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================
// 上传处理
// ============================================================

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) { selectedFile = e.target.files[0]; onFileSelected(); }
});

document.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            selectedFile = item.getAsFile();
            if (selectedFile) onFileSelected();
            return;
        }
    }
});

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault(); uploadArea.classList.remove('dragover');
    const f = e.dataTransfer.files;
    if (f.length > 0 && f[0].type.startsWith('image/')) { selectedFile = f[0]; onFileSelected(); }
    else showError('请选择有效的图像文件');
});

function onFileSelected() {
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        uploadArea.classList.add('has-file');
        // 清除所有结果
        for (const p of pipelineState.pipes) pipelineState.blobs[p.id] = null;
        pipelineState.executedUpTo = -1;
        renderPipeline();
        renderBatchPanel();
        hideError();
    };
    reader.readAsDataURL(selectedFile);
}

// ============================================================
// 管线渲染
// ============================================================

function renderPipeline() {
    pipelineFlow.innerHTML = '';
    pipelineState.pipes.forEach((pipe, idx) => {
        if (idx > 0) {
            const conn = document.createElement('div');
            conn.className = 'pipe-connector';
            pipelineFlow.appendChild(conn);
        }
        pipelineFlow.appendChild(createPipeCard(pipe, idx));
    });
    updateToolbar();
    syncTitle();
    savePipeline(pipelineState);
}

function cardState(pipe, idx) {
    const hasBlob = !!pipelineState.blobs[pipe.id];
    if (!hasBlob) return idx <= pipelineState.executedUpTo ? 'stale' : '';
    return idx > pipelineState.executedUpTo ? 'stale' : 'done';
}

function statusText(state) {
    if (state === 'done') return '已完成';
    if (state === 'stale') return '需重新执行';
    return '待执行';
}

function createPipeCard(pipe, index) {
    const reg = PIPE_REGISTRY[pipe.type];
    const cs = cardState(pipe, index);
    const hasBlob = !!pipelineState.blobs[pipe.id];

    const card = document.createElement('div');
    card.className = 'pipe-card' + (cs ? ' ' + cs : '') + (hasBlob ? ' expanded' : '');
    card.dataset.pipeId = pipe.id;
    card.dataset.index = index;

    // --- Header ---
    const header = document.createElement('div');
    header.className = 'pipe-header';
    header.innerHTML = `
        <span class="drag-handle" draggable="true">\u22EE\u22EE</span>
        <span class="pipe-label">${reg.label}</span>
        <span class="pipe-status">${statusText(cs)}</span>
        <button class="delete-btn" title="删除">\u00D7</button>
    `;

    // 展开/折叠
    header.addEventListener('click', e => {
        if (e.target.classList.contains('drag-handle') || e.target.classList.contains('delete-btn')) return;
        card.classList.toggle('expanded');
    });

    // 删除
    header.querySelector('.delete-btn').addEventListener('click', e => {
        e.stopPropagation();
        if (pipelineState.pipes.length <= 1) return;
        removePipe(pipelineState, pipe.id);
        renderPipeline();
    });

    // --- Body ---
    const body = document.createElement('div');
    body.className = 'pipe-body';

    // 参数
    const paramsDiv = document.createElement('div');
    paramsDiv.className = 'pipe-params';
    for (const pd of reg.params) paramsDiv.appendChild(createParamControl(pd, pipe));
    body.appendChild(paramsDiv);

    // 操作按钮
    const actions = document.createElement('div');
    actions.className = 'pipe-actions';

    const canExec = selectedFile && !isExecuting && (index === 0 || !!getInputBlob(pipelineState, index));
    const execBtn = document.createElement('button');
    execBtn.className = 'act-btn exec';
    execBtn.textContent = '执行此步骤';
    execBtn.disabled = !canExec;
    execBtn.addEventListener('click', () => runSingle(index));
    actions.appendChild(execBtn);

    if ((cs === 'stale' || hasBlob) && index < pipelineState.pipes.length - 1) {
        const fromBtn = document.createElement('button');
        fromBtn.className = 'act-btn exec-from';
        fromBtn.textContent = '从此步执行到底';
        fromBtn.disabled = !selectedFile || isExecuting;
        fromBtn.addEventListener('click', () => runFrom(index));
        actions.appendChild(fromBtn);
    }

    if (hasBlob) {
        const dlBtn = document.createElement('button');
        dlBtn.className = 'act-btn dl';
        dlBtn.textContent = '下载';
        dlBtn.addEventListener('click', () => triggerDownload(pipelineState.blobs[pipe.id], `${pipe.type}_result.png`));
        actions.appendChild(dlBtn);
    }
    body.appendChild(actions);

    // 结果
    if (hasBlob) body.appendChild(createResultDiv(pipe));

    card.appendChild(header);
    card.appendChild(body);
    setupDrag(card, pipe, index);
    return card;
}

// --- 参数控件 ---
function createParamControl(pd, pipe) {
    const g = document.createElement('div');
    g.className = 'param-group';

    const label = document.createElement('label');
    label.textContent = pd.label + ':';
    g.appendChild(label);

    if (pd.type === 'select') {
        const sel = document.createElement('select');
        for (const o of pd.options) {
            const opt = document.createElement('option');
            opt.value = o.value; opt.textContent = o.label;
            if (String(pipe.params[pd.key]) === String(o.value)) opt.selected = true;
            sel.appendChild(opt);
        }
        sel.addEventListener('change', () => { updateParam(pipelineState, pipe.id, pd.key, parseInt(sel.value)); renderPipeline(); });
        g.appendChild(sel);
    } else if (pd.type === 'range') {
        const input = document.createElement('input');
        input.type = 'range'; input.min = pd.min; input.max = pd.max; input.step = pd.step;
        input.value = pipe.params[pd.key];
        const val = document.createElement('span');
        val.className = 'range-val'; val.textContent = parseFloat(input.value).toFixed(1);
        input.addEventListener('input', () => { val.textContent = parseFloat(input.value).toFixed(1); });
        input.addEventListener('change', () => { updateParam(pipelineState, pipe.id, pd.key, parseFloat(input.value)); renderPipeline(); });
        g.appendChild(input); g.appendChild(val);
    } else if (pd.type === 'number') {
        const input = document.createElement('input');
        input.type = 'number'; input.min = pd.min; input.max = pd.max;
        input.value = pipe.params[pd.key]; input.style.width = '80px';
        input.addEventListener('change', () => { updateParam(pipelineState, pipe.id, pd.key, parseInt(input.value)); renderPipeline(); });
        g.appendChild(input);
    } else if (pd.type === 'color') {
        const input = document.createElement('input');
        input.type = 'color'; input.value = pipe.params[pd.key];
        input.addEventListener('change', () => { updateParam(pipelineState, pipe.id, pd.key, input.value); renderPipeline(); });
        g.appendChild(input);
    }
    return g;
}

// --- 结果区 ---
function createResultDiv(pipe) {
    const reg = PIPE_REGISTRY[pipe.type];
    const blob = pipelineState.blobs[pipe.id];
    const div = document.createElement('div');
    div.className = 'pipe-result';

    // 缩放
    const zr = document.createElement('div');
    zr.className = 'zoom-row';
    const zoomLabel = document.createElement('span');
    zoomLabel.className = 'zoom-val'; zoomLabel.textContent = '100%';
    const mkZBtn = (text, dir) => {
        const b = document.createElement('button'); b.textContent = text;
        return b;
    };
    const zOut = mkZBtn('-', -1);
    const zIn = mkZBtn('+', 1);
    const zReset = mkZBtn('重置', 0);
    zr.appendChild(zOut); zr.appendChild(zoomLabel); zr.appendChild(zIn); zr.appendChild(zReset);
    div.appendChild(zr);

    // 图片
    const vp = document.createElement('div');
    vp.className = 'result-viewport';
    const img = document.createElement('img');
    img.className = reg.resultType === 'pixelated' ? 'pixelated' : '';
    img.src = URL.createObjectURL(blob);
    vp.appendChild(img);
    div.appendChild(vp);

    // 颜色信息
    const ci = document.createElement('p');
    ci.className = 'color-info';
    img.onload = () => { ci.textContent = '颜色数: ' + countColors(img); };
    div.appendChild(ci);

    // 缩放事件
    zoomScales[pipe.id] = 1;
    const doZoom = dir => {
        if (dir === 0) zoomScales[pipe.id] = 1;
        else { zoomScales[pipe.id] *= (dir > 0 ? 2 : 0.5); zoomScales[pipe.id] = Math.max(0.25, Math.min(zoomScales[pipe.id], 32)); }
        img.style.width = (zoomScales[pipe.id] * 100) + '%';
        zoomLabel.textContent = Math.round(zoomScales[pipe.id] * 100) + '%';
    };
    zOut.addEventListener('click', () => doZoom(-1));
    zIn.addEventListener('click', () => doZoom(1));
    zReset.addEventListener('click', () => doZoom(0));

    return div;
}

// ============================================================
// 拖拽排序
// ============================================================

function setupDrag(card, pipe, index) {
    const handle = card.querySelector('.drag-handle');

    handle.addEventListener('dragstart', e => {
        e.dataTransfer.setData('text/plain', pipe.id);
        e.dataTransfer.effectAllowed = 'move';
        requestAnimationFrame(() => card.classList.add('dragging'));
    });

    handle.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        document.querySelectorAll('.pipe-card').forEach(c => c.classList.remove('drag-over-top', 'drag-over-bottom'));
    });

    card.addEventListener('dragover', e => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const rect = card.getBoundingClientRect();
        card.classList.remove('drag-over-top', 'drag-over-bottom');
        card.classList.add(e.clientY < rect.top + rect.height / 2 ? 'drag-over-top' : 'drag-over-bottom');
    });

    card.addEventListener('dragleave', () => card.classList.remove('drag-over-top', 'drag-over-bottom'));

    card.addEventListener('drop', e => {
        e.preventDefault();
        card.classList.remove('drag-over-top', 'drag-over-bottom');
        const draggedId = e.dataTransfer.getData('text/plain');
        if (draggedId === pipe.id) return;
        const fromIdx = pipelineState.pipes.findIndex(p => p.id === draggedId);
        if (fromIdx === -1) return;
        const rect = card.getBoundingClientRect();
        const insertAfter = e.clientY >= rect.top + rect.height / 2;
        const toIdx = insertAfter ? index + 1 : index;
        movePipe(pipelineState, fromIdx, toIdx);
        renderPipeline();
    });
}

// ============================================================
// 管线执行
// ============================================================

async function executePipe(index) {
    const pipe = pipelineState.pipes[index];
    const reg = PIPE_REGISTRY[pipe.type];
    const inputBlob = index === 0 ? selectedFile : getInputBlob(pipelineState, index);
    if (!inputBlob) throw new Error('缺少输入图像');

    // 更新卡片 UI
    const card = pipelineFlow.querySelector(`[data-pipe-id="${pipe.id}"]`);
    if (card) {
        card.className = 'pipe-card running expanded';
        card.querySelector('.pipe-status').textContent = '执行中...';
    }
    showLoading(reg.label + ' 处理中...');

    try {
        const fd = new FormData();
        fd.append('image', inputBlob, index === 0 ? selectedFile.name : 'input.png');
        for (const [k, v] of Object.entries(pipe.params)) fd.append(k, v);

        const res = await fetch(reg.endpoint, { method: 'POST', body: fd });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: '处理失败' }));
            throw new Error(err.detail || '处理失败');
        }

        pipelineState.blobs[pipe.id] = await res.blob();
        pipelineState.executedUpTo = Math.max(pipelineState.executedUpTo, index);
    } finally {
        hideLoading();
    }
}

async function runSingle(index) {
    hideError(); isExecuting = true;
    try { await executePipe(index); }
    catch (e) { showError(e.message); }
    finally { isExecuting = false; renderPipeline(); }
}

async function runAll() {
    if (!selectedFile || isExecuting) return;
    hideError(); isExecuting = true;
    try {
        for (let i = 0; i < pipelineState.pipes.length; i++) await executePipe(i);
    } catch (e) { showError(e.message); }
    finally { isExecuting = false; renderPipeline(); }
}

async function runFrom(startIndex) {
    if (!selectedFile || isExecuting) return;
    hideError(); isExecuting = true;
    try {
        for (let i = startIndex; i < pipelineState.pipes.length; i++) await executePipe(i);
    } catch (e) { showError(e.message); }
    finally { isExecuting = false; renderPipeline(); }
}

// ============================================================
// 工具栏
// ============================================================

function updateToolbar() {
    executeAllBtn.disabled = !selectedFile || isExecuting;
}

function populateMenus() {
    // 添加步骤菜单
    const addMenu = $('addPipeMenu');
    for (const [type, reg] of Object.entries(PIPE_REGISTRY)) {
        const btn = document.createElement('button');
        btn.className = 'dropdown-item'; btn.textContent = reg.label;
        btn.addEventListener('mousedown', e => {
            e.preventDefault();
            e.stopPropagation();
            addPipe(pipelineState, type);
            renderPipeline();
            closeAllDropdowns();
        });
        addMenu.appendChild(btn);
    }

    // 预设菜单
    const presetMenu = $('presetMenu');
    for (const [key, preset] of Object.entries(PIPELINE_PRESETS)) {
        const btn = document.createElement('button');
        btn.className = 'dropdown-item'; btn.textContent = preset.label;
        btn.addEventListener('mousedown', e => {
            e.preventDefault();
            e.stopPropagation();
            pipelineState = createPipelineState(preset.pipes);
            pipelineName = preset.label;
            renderPipeline();
            closeAllDropdowns();
        });
        presetMenu.appendChild(btn);
    }
}

function toggleDropdown(menuId) {
    const menu = $(menuId);
    const open = menu.classList.contains('show');
    closeAllDropdowns();
    if (!open) menu.classList.add('show');
}

function closeAllDropdowns() {
    document.querySelectorAll('.dropdown-menu').forEach(m => m.classList.remove('show'));
}

$('addPipeBtn').addEventListener('mousedown', e => { e.preventDefault(); e.stopPropagation(); toggleDropdown('addPipeMenu'); });
$('presetBtn').addEventListener('mousedown', e => { e.preventDefault(); e.stopPropagation(); toggleDropdown('presetMenu'); });
document.addEventListener('mousedown', e => { if (!e.target.closest('.dropdown')) closeAllDropdowns(); });

executeAllBtn.addEventListener('click', runAll);

// ============================================================
// 保存/加载配置
// ============================================================

const SAVED_CONFIGS_KEY = 'pixelization_saved_configs';

function loadSavedConfigs() {
    try { return JSON.parse(localStorage.getItem(SAVED_CONFIGS_KEY)) || {}; }
    catch (e) { return {}; }
}

function saveSavedConfigs(configs) {
    try { localStorage.setItem(SAVED_CONFIGS_KEY, JSON.stringify(configs)); } catch (e) {}
}

const saveForm = $('saveForm');
const saveConfigName = $('saveConfigName');

$('saveConfigBtn').addEventListener('click', () => {
    saveForm.classList.add('show');
    $('saveConfigBtn').style.display = 'none';
    saveConfigName.value = '';
    saveConfigName.focus();
});

$('saveCancelBtn').addEventListener('click', () => {
    saveForm.classList.remove('show');
    $('saveConfigBtn').style.display = '';
});

function doSaveConfig() {
    const name = saveConfigName.value.trim();
    if (!name) { saveConfigName.focus(); return; }
    const configs = loadSavedConfigs();
    configs[name] = pipelineState.pipes.map(p => ({ type: p.type, params: p.params }));
    saveSavedConfigs(configs);
    saveForm.classList.remove('show');
    $('saveConfigBtn').style.display = '';
    populateSavedConfigMenu();
}

$('saveConfirmBtn').addEventListener('click', doSaveConfig);
saveConfigName.addEventListener('keydown', e => { if (e.key === 'Enter') doSaveConfig(); });

// "我的配置" 下拉
function populateSavedConfigMenu() {
    const menu = $('savedConfigMenu');
    menu.innerHTML = '';
    const configs = loadSavedConfigs();
    const names = Object.keys(configs);
    if (names.length === 0) {
        const hint = document.createElement('div');
        hint.style.cssText = 'padding:10px 16px;font-size:12px;color:#999;';
        hint.textContent = '暂无保存的配置';
        menu.appendChild(hint);
        return;
    }
    for (const name of names) {
        const item = document.createElement('div');
        item.className = 'dropdown-item saved';
        const label = document.createElement('span');
        label.textContent = name;
        label.style.cursor = 'pointer';
        label.style.flex = '1';
        label.addEventListener('mousedown', e => {
            e.preventDefault(); e.stopPropagation();
            const pipes = configs[name];
            pipelineState = { pipes: pipes.map(p => ({ type: p.type, id: nextPipeId(), params: { ...defaultParams(p.type), ...p.params } })), blobs: {}, executedUpTo: -1 };
            for (const p of pipelineState.pipes) pipelineState.blobs[p.id] = null;
            pipelineName = name;
            renderPipeline();
            closeAllDropdowns();
        });
        const del = document.createElement('button');
        del.className = 'del-config'; del.textContent = '\u00D7'; del.title = '删除';
        del.addEventListener('mousedown', e => {
            e.preventDefault(); e.stopPropagation();
            const c = loadSavedConfigs();
            delete c[name];
            saveSavedConfigs(c);
            populateSavedConfigMenu();
        });
        item.appendChild(label);
        item.appendChild(del);
        menu.appendChild(item);
    }
}

$('savedConfigBtn').addEventListener('mousedown', e => {
    e.preventDefault(); e.stopPropagation();
    populateSavedConfigMenu();
    toggleDropdown('savedConfigMenu');
});

// ============================================================
// 管线名称
// ============================================================

const pipelineNameInput = $('pipelineName');

pipelineNameInput.addEventListener('change', () => {
    pipelineName = pipelineNameInput.value.trim() || '未命名管线';
    pipelineNameInput.value = pipelineName;
    savePipeline(pipelineState);
});

function syncTitle() {
    pipelineNameInput.value = pipelineName;
}

// ============================================================
// 导入/导出全部配置 JSON
// ============================================================

$('exportBtn').addEventListener('click', () => {
    const configs = loadSavedConfigs();
    if (Object.keys(configs).length === 0) { showError('暂无已保存的配置'); return; }
    const blob = new Blob([JSON.stringify(configs, null, 2)], { type: 'application/json' });
    triggerDownload(blob, 'pixelization_configs.json');
});

const importFileInput = $('importFileInput');

$('importBtn').addEventListener('click', () => importFileInput.click());

importFileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
        try {
            const data = JSON.parse(ev.target.result);
            if (typeof data !== 'object' || Array.isArray(data)) throw new Error('格式错误，应为 { "名称": [步骤] }');
            const existing = loadSavedConfigs();
            let count = 0;
            for (const [name, pipes] of Object.entries(data)) {
                if (!Array.isArray(pipes)) continue;
                const valid = pipes.filter(p => p.type && PIPE_REGISTRY[p.type]);
                if (valid.length === 0) continue;
                existing[name] = valid.map(p => ({ type: p.type, params: p.params }));
                count++;
            }
            if (count === 0) throw new Error('文件中没有有效配置');
            saveSavedConfigs(existing);
            populateSavedConfigMenu();
            hideError();
        } catch (err) {
            showError('导入失败: ' + err.message);
        }
    };
    reader.readAsText(file);
    importFileInput.value = '';
});

$('resetBtn').addEventListener('click', () => {
    localStorage.removeItem(STORAGE_KEY);
    pipelineState = createPipelineState(PIPELINE_PRESETS['default'].pipes);
    pipelineName = '默认管线';
    renderPipeline();
});

// ============================================================
// Batch 导入/导出
// ============================================================

function exportBatch(batch) {
    const json = JSON.stringify(batch, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    triggerDownload(blob, (batch.name || 'batch') + '.json');
}

function importBatchFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => {
            try {
                const data = JSON.parse(e.target.result);
                const batch = validateBatchJSON(data);
                if (!batch) throw new Error('JSON 格式不符合 batch 配置规范');
                // 补全 params 缺省值
                for (const cfg of batch.configs) {
                    cfg.pipeline = validatePipelineSteps(cfg.pipeline);
                }
                resolve(batch);
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('文件读取失败'));
        reader.readAsText(file);
    });
}

// 全局 batches 列表
let batches = loadBatches();

// ============================================================
// Batch 面板
// ============================================================

let currentBatchIndex = -1;
let expandedConfigIdx = -1;
let batchExecuting = false;
let batchResults = [];
let collageZoom = 1;

const batchSelect = $('batchSelect');
const batchNameEl = $('batchName');
const batchTileSizeEl = $('batchTileSize');
const batchMetaEl = $('batchMeta');
const batchToolbarEl = $('batchToolbar');
const batchConfigsEl = $('batchConfigs');
const batchPreviewEl = $('batchPreview');
const collageCanvasEl = $('collageCanvas');

function getCurrentBatch() {
    return (currentBatchIndex >= 0 && currentBatchIndex < batches.length) ? batches[currentBatchIndex] : null;
}

function pipelineSummary(pipeline) {
    if (!pipeline || pipeline.length === 0) return '(空)';
    return pipeline.map(s => {
        const reg = PIPE_REGISTRY[s.type];
        if (!reg) return s.type;
        const vals = Object.values(s.params).join(', ');
        return reg.label + (vals ? '(' + vals + ')' : '');
    }).join(' → ');
}

function renderBatchPanel() {
    batchSelect.innerHTML = '';
    if (batches.length === 0) {
        const opt = document.createElement('option');
        opt.textContent = '(无 Batch)';
        batchSelect.appendChild(opt);
        currentBatchIndex = -1;
    } else {
        batches.forEach((b, i) => {
            const opt = document.createElement('option');
            opt.value = i; opt.textContent = b.name;
            opt.selected = i === currentBatchIndex;
            batchSelect.appendChild(opt);
        });
        if (currentBatchIndex < 0 || currentBatchIndex >= batches.length) currentBatchIndex = 0;
        batchSelect.value = String(currentBatchIndex);
    }
    const batch = getCurrentBatch();
    batchMetaEl.style.display = batch ? '' : 'none';
    batchToolbarEl.style.display = batch ? '' : 'none';
    if (!batch) {
        batchConfigsEl.innerHTML = '<p class="batch-empty">点击「新建」创建一个 Batch 配置</p>';
        batchPreviewEl.style.display = 'none';
        return;
    }
    batchNameEl.value = batch.name;
    batchTileSizeEl.value = batch.tile_size;
    $('executeBatchBtn').disabled = !selectedFile || batchExecuting;
    renderBatchConfigs();
}

function renderBatchConfigs() {
    const batch = getCurrentBatch();
    if (!batch) return;
    batchConfigsEl.innerHTML = '';
    if (batch.configs.length === 0) {
        batchConfigsEl.innerHTML = '<p class="batch-empty">点击「+ 添加配置」添加处理配置</p>';
        return;
    }
    batch.configs.forEach((cfg, idx) => {
        if (idx > 0) {
            const conn = document.createElement('div');
            conn.className = 'pipe-connector';
            batchConfigsEl.appendChild(conn);
        }
        batchConfigsEl.appendChild(createBatchConfigCard(cfg, idx));
    });
}

function createBatchConfigCard(config, index) {
    const expanded = expandedConfigIdx === index;
    const card = document.createElement('div');
    card.className = 'batch-config-card' + (expanded ? ' expanded' : '');
    card.dataset.configIndex = index;

    // Header
    const header = document.createElement('div');
    header.className = 'batch-config-header';
    header.innerHTML = `
        <span class="drag-handle" draggable="true">\u22EE\u22EE</span>
        <span class="config-idx">${index + 1}.</span>
        <span class="config-name">${config.name}</span>
        <span class="config-summary">${pipelineSummary(config.pipeline)}</span>
        <button class="delete-btn" title="删除">\u00D7</button>
    `;
    header.addEventListener('click', e => {
        if (e.target.classList.contains('drag-handle') || e.target.classList.contains('delete-btn')) return;
        expandedConfigIdx = expandedConfigIdx === index ? -1 : index;
        renderBatchConfigs();
    });
    header.querySelector('.delete-btn').addEventListener('click', e => {
        e.stopPropagation();
        const batch = getCurrentBatch();
        removeBatchConfig(batch, index);
        if (expandedConfigIdx === index) expandedConfigIdx = -1;
        else if (expandedConfigIdx > index) expandedConfigIdx--;
        saveBatches(batches);
        renderBatchConfigs();
    });
    card.appendChild(header);
    setupBatchConfigDrag(card, index);

    if (expanded) {
        const body = document.createElement('div');
        body.className = 'batch-config-body';
        // Name edit
        const nameRow = document.createElement('div');
        nameRow.className = 'config-name-edit';
        const nameLabel = document.createElement('label');
        nameLabel.textContent = '名称:';
        nameRow.appendChild(nameLabel);
        const nameInput = document.createElement('input');
        nameInput.type = 'text'; nameInput.value = config.name; nameInput.maxLength = 30;
        nameInput.addEventListener('change', () => {
            config.name = nameInput.value.trim() || ('配置 ' + (index + 1));
            saveBatches(batches);
            renderBatchConfigs();
        });
        nameRow.appendChild(nameInput);
        body.appendChild(nameRow);
        // Steps
        const stepsDiv = document.createElement('div');
        stepsDiv.className = 'config-steps';
        config.pipeline.forEach((step, si) => stepsDiv.appendChild(createBatchStepRow(config, index, step, si)));
        body.appendChild(stepsDiv);
        // Add step dropdown
        const addRow = document.createElement('div');
        addRow.className = 'add-step-row';
        const dd = document.createElement('div');
        dd.className = 'dropdown';
        const ddBtn = document.createElement('button');
        ddBtn.className = 'toolbar-btn'; ddBtn.textContent = '+ 添加步骤';
        const ddMenu = document.createElement('div');
        ddMenu.className = 'dropdown-menu';
        for (const [type, reg] of Object.entries(PIPE_REGISTRY)) {
            const item = document.createElement('button');
            item.className = 'dropdown-item'; item.textContent = reg.label;
            item.addEventListener('mousedown', ev => {
                ev.preventDefault(); ev.stopPropagation();
                config.pipeline.push({ type, params: defaultParams(type) });
                saveBatches(batches);
                renderBatchConfigs();
            });
            ddMenu.appendChild(item);
        }
        ddBtn.addEventListener('mousedown', ev => {
            ev.preventDefault(); ev.stopPropagation();
            closeAllDropdowns();
            ddMenu.classList.toggle('show');
        });
        dd.appendChild(ddBtn); dd.appendChild(ddMenu);
        addRow.appendChild(dd);
        body.appendChild(addRow);
        card.appendChild(body);
    }
    return card;
}

function createBatchStepRow(config, configIdx, step, stepIdx) {
    const reg = PIPE_REGISTRY[step.type];
    const row = document.createElement('div');
    row.className = 'config-step-row';
    const typeLabel = document.createElement('span');
    typeLabel.className = 'step-type';
    typeLabel.textContent = reg ? reg.label : step.type;
    row.appendChild(typeLabel);

    const paramsEl = document.createElement('span');
    paramsEl.className = 'step-params';
    const doSave = () => { saveBatches(batches); updateConfigSummary(configIdx); };
    for (const pd of (reg ? reg.params : [])) {
        paramsEl.appendChild(createBatchParamCtrl(pd, step, doSave));
    }
    row.appendChild(paramsEl);

    // Move up/down
    const mkMove = (text, delta) => {
        const b = document.createElement('button');
        b.className = 'step-move'; b.textContent = text;
        const target = stepIdx + delta;
        b.disabled = target < 0 || target >= config.pipeline.length;
        b.addEventListener('click', () => {
            [config.pipeline[stepIdx], config.pipeline[target]] = [config.pipeline[target], config.pipeline[stepIdx]];
            saveBatches(batches);
            renderBatchConfigs();
        });
        return b;
    };
    row.appendChild(mkMove('\u2191', -1));
    row.appendChild(mkMove('\u2193', 1));

    const del = document.createElement('button');
    del.className = 'delete-btn'; del.textContent = '\u00D7';
    del.addEventListener('click', () => {
        config.pipeline.splice(stepIdx, 1);
        saveBatches(batches);
        renderBatchConfigs();
    });
    row.appendChild(del);
    return row;
}

function createBatchParamCtrl(pd, step, onSave) {
    const g = document.createElement('span');
    g.className = 'param-group';
    const label = document.createElement('label');
    label.textContent = pd.label + ':';
    g.appendChild(label);

    if (pd.type === 'select') {
        const sel = document.createElement('select');
        for (const o of pd.options) {
            const opt = document.createElement('option');
            opt.value = o.value; opt.textContent = o.label;
            if (String(step.params[pd.key]) === String(o.value)) opt.selected = true;
            sel.appendChild(opt);
        }
        sel.addEventListener('change', () => { step.params[pd.key] = parseInt(sel.value); onSave(); });
        g.appendChild(sel);
    } else if (pd.type === 'range') {
        const input = document.createElement('input');
        input.type = 'range'; input.min = pd.min; input.max = pd.max; input.step = pd.step;
        input.value = step.params[pd.key];
        const val = document.createElement('span');
        val.className = 'range-val'; val.textContent = parseFloat(input.value).toFixed(1);
        input.addEventListener('input', () => val.textContent = parseFloat(input.value).toFixed(1));
        input.addEventListener('change', () => { step.params[pd.key] = parseFloat(input.value); onSave(); });
        g.appendChild(input); g.appendChild(val);
    } else if (pd.type === 'number') {
        const input = document.createElement('input');
        input.type = 'number'; input.min = pd.min; input.max = pd.max;
        input.value = step.params[pd.key]; input.style.width = '70px';
        input.addEventListener('change', () => { step.params[pd.key] = parseInt(input.value); onSave(); });
        g.appendChild(input);
    } else if (pd.type === 'color') {
        const input = document.createElement('input');
        input.type = 'color'; input.value = step.params[pd.key];
        input.addEventListener('change', () => { step.params[pd.key] = input.value; onSave(); });
        g.appendChild(input);
    }
    return g;
}

function updateConfigSummary(configIdx) {
    const batch = getCurrentBatch();
    if (!batch || configIdx >= batch.configs.length) return;
    const card = batchConfigsEl.querySelector('[data-config-index="' + configIdx + '"]');
    if (!card) return;
    const el = card.querySelector('.config-summary');
    if (el) el.textContent = pipelineSummary(batch.configs[configIdx].pipeline);
}

// Batch config drag-and-drop
function setupBatchConfigDrag(card, index) {
    const handle = card.querySelector('.drag-handle');
    handle.addEventListener('dragstart', e => {
        e.dataTransfer.setData('batch-config', String(index));
        e.dataTransfer.effectAllowed = 'move';
        requestAnimationFrame(() => card.classList.add('dragging'));
    });
    handle.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        document.querySelectorAll('.batch-config-card').forEach(c => c.classList.remove('drag-over-top', 'drag-over-bottom'));
    });
    card.addEventListener('dragover', e => {
        if (!e.dataTransfer.types.includes('batch-config')) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const rect = card.getBoundingClientRect();
        card.classList.remove('drag-over-top', 'drag-over-bottom');
        card.classList.add(e.clientY < rect.top + rect.height / 2 ? 'drag-over-top' : 'drag-over-bottom');
    });
    card.addEventListener('dragleave', () => card.classList.remove('drag-over-top', 'drag-over-bottom'));
    card.addEventListener('drop', e => {
        e.preventDefault();
        card.classList.remove('drag-over-top', 'drag-over-bottom');
        if (!e.dataTransfer.types.includes('batch-config')) return;
        const fromIdx = parseInt(e.dataTransfer.getData('batch-config'));
        if (isNaN(fromIdx) || fromIdx === index) return;
        const batch = getCurrentBatch();
        const rect = card.getBoundingClientRect();
        const toIdx = e.clientY >= rect.top + rect.height / 2 ? index + 1 : index;
        moveBatchConfig(batch, fromIdx, toIdx);
        if (expandedConfigIdx === fromIdx) expandedConfigIdx = fromIdx < toIdx ? toIdx - 1 : toIdx;
        saveBatches(batches);
        renderBatchConfigs();
    });
}

// --- Batch toolbar ---
function populateAddBatchConfigMenu() {
    const menu = $('addBatchConfigMenu');
    menu.innerHTML = '';
    // From current pipeline
    const fromCurrent = document.createElement('button');
    fromCurrent.className = 'dropdown-item'; fromCurrent.textContent = '从当前管线';
    fromCurrent.addEventListener('mousedown', e => {
        e.preventDefault(); e.stopPropagation();
        const batch = getCurrentBatch();
        if (!batch) return;
        addBatchConfig(batch, null, pipelineToSteps(pipelineState));
        saveBatches(batches);
        renderBatchConfigs();
        closeAllDropdowns();
    });
    menu.appendChild(fromCurrent);
    // Saved configs
    const savedConfigs = loadSavedConfigs();
    const names = Object.keys(savedConfigs);
    if (names.length > 0) {
        const sep = document.createElement('div');
        sep.style.cssText = 'border-top:1px solid #f0f0f0;margin:4px 0;';
        menu.appendChild(sep);
        for (const name of names) {
            const item = document.createElement('button');
            item.className = 'dropdown-item'; item.textContent = name;
            item.addEventListener('mousedown', e => {
                e.preventDefault(); e.stopPropagation();
                const batch = getCurrentBatch();
                if (!batch) return;
                addBatchConfig(batch, name, validatePipelineSteps(savedConfigs[name]));
                saveBatches(batches);
                renderBatchConfigs();
                closeAllDropdowns();
            });
            menu.appendChild(item);
        }
    }
    const sep2 = document.createElement('div');
    sep2.style.cssText = 'border-top:1px solid #f0f0f0;margin:4px 0;';
    menu.appendChild(sep2);
    const empty = document.createElement('button');
    empty.className = 'dropdown-item'; empty.textContent = '空白配置';
    empty.addEventListener('mousedown', e => {
        e.preventDefault(); e.stopPropagation();
        const batch = getCurrentBatch();
        if (!batch) return;
        addBatchConfig(batch, null, []);
        saveBatches(batches);
        renderBatchConfigs();
        closeAllDropdowns();
    });
    menu.appendChild(empty);
}

// --- Batch execution ---
async function executeBatch() {
    const batch = getCurrentBatch();
    if (!batch || !selectedFile || batchExecuting) return;
    batchExecuting = true;
    hideError();
    $('executeBatchBtn').disabled = true;
    const results = [];
    for (let i = 0; i < batch.configs.length; i++) {
        const cfg = batch.configs[i];
        try {
            showLoading('Batch ' + (i + 1) + '/' + batch.configs.length + ': ' + cfg.name);
            const blob = await executeBatchPipeline(cfg.pipeline);
            const img = new Image();
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = () => reject(new Error('图像加载失败'));
                img.src = URL.createObjectURL(blob);
            });
            results.push({ name: cfg.name, blob, img });
        } catch (e) {
            results.push({ name: cfg.name, error: e.message });
        }
    }
    hideLoading();
    batchExecuting = false;
    batchResults = results;
    $('executeBatchBtn').disabled = false;
    const success = results.filter(r => !r.error);
    if (success.length > 0) {
        generateCollage(success, batch.tile_size);
        batchPreviewEl.style.display = '';
    }
    if (results.some(r => r.error)) {
        showError(results.filter(r => r.error).map(r => r.name + ': ' + r.error).join('; '));
    }
}

async function executeBatchPipeline(pipeline) {
    let currentBlob = selectedFile;
    if (pipeline.length === 0) return currentBlob;
    for (const step of pipeline) {
        const reg = PIPE_REGISTRY[step.type];
        if (!reg) throw new Error('未知步骤: ' + step.type);
        const fd = new FormData();
        fd.append('image', currentBlob, 'input.png');
        for (const [k, v] of Object.entries(step.params)) fd.append(k, v);
        const res = await fetch(reg.endpoint, { method: 'POST', body: fd });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: '处理失败' }));
            throw new Error(err.detail || '处理失败');
        }
        currentBlob = await res.blob();
    }
    return currentBlob;
}

// --- Tile collage ---
function generateCollage(results, tileSize) {
    const items = results.map(r => {
        const gw = Math.ceil(r.img.naturalWidth / tileSize);
        const gh = Math.ceil(r.img.naturalHeight / tileSize);
        return { ...r, gw, gh };
    }).sort((a, b) => (b.gw * b.gh) - (a.gw * a.gh));

    if (items.length === 0) return;

    // Greedy bin packing on tile grid
    const occupied = new Set();
    const ok = (x, y, gw, gh) => {
        for (let dy = 0; dy < gh; dy++)
            for (let dx = 0; dx < gw; dx++)
                if (occupied.has((x + dx) + ',' + (y + dy))) return false;
        return true;
    };
    const mark = (x, y, gw, gh) => {
        for (let dy = 0; dy < gh; dy++)
            for (let dx = 0; dx < gw; dx++)
                occupied.add((x + dx) + ',' + (y + dy));
    };

    const totalCells = items.reduce((s, g) => s + g.gw * g.gh, 0);
    const targetCols = Math.max(1, Math.ceil(Math.sqrt(totalCells)));
    const positions = [];

    for (const item of items) {
        let placed = false;
        for (let gy = 0; !placed; gy++) {
            for (let gx = 0; gx <= targetCols; gx++) {
                if (ok(gx, gy, item.gw, item.gh)) {
                    mark(gx, gy, item.gw, item.gh);
                    positions.push({ x: gx * tileSize, y: gy * tileSize, item });
                    placed = true;
                    break;
                }
            }
        }
    }

    let cw = 0, ch = 0;
    for (const p of positions) {
        cw = Math.max(cw, p.x + p.item.img.naturalWidth);
        ch = Math.max(ch, p.y + p.item.img.naturalHeight);
    }

    collageCanvasEl.width = cw;
    collageCanvasEl.height = ch;
    const ctx = collageCanvasEl.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
    for (const p of positions) ctx.drawImage(p.item.img, p.x, p.y);

    collageZoom = 1;
    $('collageZoomVal').textContent = '100%';
    collageCanvasEl.style.width = cw + 'px';
}

// --- Batch event bindings ---
batchSelect.addEventListener('change', () => {
    currentBatchIndex = parseInt(batchSelect.value);
    expandedConfigIdx = -1;
    batchPreviewEl.style.display = 'none';
    renderBatchPanel();
});

$('newBatchBtn').addEventListener('click', () => {
    const batch = createBatch('Batch ' + (batches.length + 1), 16);
    batches.push(batch);
    currentBatchIndex = batches.length - 1;
    expandedConfigIdx = -1;
    saveBatches(batches);
    renderBatchPanel();
});

$('deleteBatchBtn').addEventListener('click', () => {
    if (batches.length === 0) return;
    batches.splice(currentBatchIndex, 1);
    if (currentBatchIndex >= batches.length) currentBatchIndex = batches.length - 1;
    expandedConfigIdx = -1;
    saveBatches(batches);
    batchPreviewEl.style.display = 'none';
    renderBatchPanel();
});

batchNameEl.addEventListener('change', () => {
    const batch = getCurrentBatch();
    if (!batch) return;
    batch.name = batchNameEl.value.trim() || '未命名 Batch';
    batchNameEl.value = batch.name;
    saveBatches(batches);
    batchSelect.options[currentBatchIndex].textContent = batch.name;
});

batchTileSizeEl.addEventListener('change', () => {
    const batch = getCurrentBatch();
    if (!batch) return;
    batch.tile_size = Math.max(8, Math.min(64, parseInt(batchTileSizeEl.value) || 16));
    batchTileSizeEl.value = batch.tile_size;
    saveBatches(batches);
});

$('addBatchConfigBtn').addEventListener('mousedown', e => {
    e.preventDefault(); e.stopPropagation();
    populateAddBatchConfigMenu();
    toggleDropdown('addBatchConfigMenu');
});

$('importBatchBtn').addEventListener('click', () => $('importBatchFileInput').click());
$('importBatchFileInput').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const batch = await importBatchFromFile(file);
        batches.push(batch);
        currentBatchIndex = batches.length - 1;
        saveBatches(batches);
        renderBatchPanel();
        hideError();
    } catch (err) { showError('导入失败: ' + err.message); }
    e.target.value = '';
});

$('exportBatchBtn').addEventListener('click', () => {
    const batch = getCurrentBatch();
    if (!batch) { showError('请先选择一个 Batch'); return; }
    exportBatch(batch);
});

$('executeBatchBtn').addEventListener('click', executeBatch);

$('downloadCollageBtn').addEventListener('click', () => {
    collageCanvasEl.toBlob(blob => {
        triggerDownload(blob, (getCurrentBatch()?.name || 'collage') + '_tiled.png');
    }, 'image/png');
});

// Collage zoom
const doCollageZoom = dir => {
    if (dir === 0) collageZoom = 1;
    else { collageZoom *= (dir > 0 ? 2 : 0.5); collageZoom = Math.max(0.25, Math.min(collageZoom, 16)); }
    collageCanvasEl.style.width = (collageCanvasEl.width * collageZoom) + 'px';
    $('collageZoomVal').textContent = Math.round(collageZoom * 100) + '%';
};
$('collageZoomIn').addEventListener('click', () => doCollageZoom(1));
$('collageZoomOut').addEventListener('click', () => doCollageZoom(-1));
$('collageZoomReset').addEventListener('click', () => doCollageZoom(0));

// ============================================================
// 初始化
// ============================================================

populateMenus();
renderPipeline();
renderBatchPanel();
