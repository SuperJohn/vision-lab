// Vision Lab client: live stream rendering plus the zones, timeline and model screens.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtTime = (ts) => VLTime.clock(ts);   // shared 12-hour formatter (timefmt.js)
const ZONE_COLORS = ['#F5A524', '#5CC8FF', '#9BE15D', '#FF8CC0', '#C4AEFF', '#FFD166'];
const SKELETON = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers = { 'Content-Type': 'application/json' };
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

let state = { settings: null, zones: [], rules: [], classes: [], vlms: [], llms: [], cameras: [] };
let meta = null;

// reload the page when the server restarts (boot ID changes), so it never runs stale code.
let bootId = null;
function checkBoot(id) {
  if (!id) return;
  if (bootId === null) bootId = id;
  else if (id !== bootId) location.reload();
}

// ---------------- tabs ----------------
function show(view) {
  document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.view').forEach((v) => (v.hidden = v.id !== `view-${view}`));
  if (view === 'zones') loadPausedFrame();
  if (view === 'timeline') loadTimeline();
  if (view === 'models') loadModels();
  if (view === 'screens') loadScreens();
  if (view === 'voice' && window.vlabLoadVoiceLog) window.vlabLoadVoiceLog();
}
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => show(b.dataset.view)));
document.querySelectorAll('[data-goto]').forEach((b) => b.addEventListener('click', () => show(b.dataset.goto)));

// ---------------- state & settings ----------------
async function loadState() {
  state = await api('/api/state');
  const s = state.settings;
  $('#camera').innerHTML = state.cameras.map((c) => `<option value="${c.index}">${esc(c.name)}</option>`).join('');
  $('#camera').value = s.camera;
  $('#detector').value = s.detector;
  $('#world-classes').value = s.world_classes;
  $('#world-wrap').hidden = !s.detector.includes('world');
  const vlms = state.vlms.includes(s.describe_model) ? state.vlms : [s.describe_model, ...state.vlms];
  $('#describe-model').innerHTML = vlms.map((m) => `<option>${esc(m)}</option>`).join('');
  $('#describe-model').value = s.describe_model;
  const llms = state.llms.includes(s.summary_model) ? state.llms : [s.summary_model, ...state.llms];
  $('#summary-model').innerHTML = llms.map((m) => `<option>${esc(m)}</option>`).join('');
  $('#summary-model').value = s.summary_model;
  $('#describe-every').value = s.describe_every;
  $('#conf').value = s.conf;
  $('#conf-val').textContent = Number(s.conf).toFixed(2);
  renderStages();
  renderZoneList();
  renderRules();
}

async function setSettings(patch) {
  state.settings = await api('/api/settings', { method: 'POST', body: patch });
  $('#world-wrap').hidden = !state.settings.detector.includes('world');
  renderStages();
}
$('#camera').addEventListener('change', (e) => setSettings({ camera: Number(e.target.value) }));
$('#detector').addEventListener('change', (e) => setSettings({ detector: e.target.value }));
$('#world-classes').addEventListener('change', (e) => setSettings({ world_classes: e.target.value }));
$('#describe-model').addEventListener('change', (e) => setSettings({ describe_model: e.target.value }));
$('#summary-model').addEventListener('change', (e) => setSettings({ summary_model: e.target.value }));
$('#describe-every').addEventListener('change', (e) => setSettings({ describe_every: Math.max(2, Number(e.target.value) || 5) }));
$('#conf').addEventListener('input', (e) => { $('#conf-val').textContent = Number(e.target.value).toFixed(2); });
$('#conf').addEventListener('change', (e) => setSettings({ conf: Number(e.target.value) }));

const STAGES = [
  { key: 'capture', name: 'Capture', hw: 'CPU' },
  { key: 'detect', name: 'Detect', hw: 'GPU' },
  { key: 'track', name: 'Track', hw: 'CPU' },
  { key: 'segment', name: 'Segment', hw: 'GPU' },
  { key: 'pose', name: 'Pose', hw: 'GPU' },
  { key: 'depth', name: 'Depth', hw: 'GPU' },
  { key: 'describe', name: 'Describe', hw: 'GPU' },
];

function stageInfo(key) {
  const s = state.settings;
  const st = s.stages;
  const world = s.detector.includes('world');
  const n = meta?.objects?.length ?? 0;
  const fps = meta?.fps || {};
  switch (key) {
    case 'capture': return [true, `${meta?.w ? `${meta.w}×${meta.h}` : '—'} · ${fps.camera ?? 0} fps`];
    case 'detect': return [st.detect, `${s.detector} · ${n} objects · ${fps.pipeline ?? 0} fps`];
    case 'track': return [st.detect, `ByteTrack · ${new Set((meta?.objects || []).map((o) => o.id)).size} IDs`];
    case 'segment': return [st.segment && !world, world ? 'Not available with YOLO-World' : `${s.detector}-seg`];
    case 'pose': return [st.pose, 'YOLO11s-pose · 17 keypoints'];
    case 'depth': return [st.depth, `Depth Anything V2 Small${st.depth ? '' : ' · off'}`];
    case 'describe': return [st.describe, `${s.describe_model} · every ${s.describe_every} s`];
  }
}

function renderStages() {
  if (!state.settings) return;
  $('#stages').innerHTML = STAGES.map((g) => {
    const [on, sub] = stageInfo(g.key);
    const fixed = g.key === 'capture' || g.key === 'track';
    return `<button type="button" class="stage-row ${on ? '' : 'off'}" data-stage="${g.key}" ${fixed ? 'disabled' : ''}
      aria-pressed="${on}" title="${fixed ? '' : 'Click to turn on or off'}">
      <span class="dot ${on ? 'ok' : ''}"></span>
      <span class="grow" style="min-width:0"><div class="name">${g.name}</div><div class="sub">${esc(sub)}</div></span>
      <span class="hw ${g.hw}">${g.hw}</span></button>`;
  }).join('');
}
$('#stages').addEventListener('click', (e) => {
  const b = e.target.closest('[data-stage]');
  if (!b || b.disabled) return;
  const k = b.dataset.stage;
  setSettings({ stages: { [k]: !state.settings.stages[k] } });
});

// ---------------- live stream ----------------
const layers = { boxes: true, masks: true, pose: true, tracks: true, zones: true, depth: true };
function renderLayers() {
  $('#layers').innerHTML = Object.keys(layers).map((k) =>
    `<button type="button" class="btn layer ${layers[k] ? 'on' : ''}" data-layer="${k}" aria-pressed="${layers[k]}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('');
}
$('#layers').addEventListener('click', (e) => {
  const b = e.target.closest('[data-layer]');
  if (!b) return;
  layers[b.dataset.layer] = !layers[b.dataset.layer];
  renderLayers();
});
renderLayers();

const live = $('#live');
const lctx = live.getContext('2d');
const depthImg = new Image();
let depthSrc = null;
let pendingMeta = null;
let lastEventsRev = -1;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'blob';
  ws.onmessage = async (ev) => {
    if (typeof ev.data === 'string') {
      pendingMeta = JSON.parse(ev.data);
      onMeta(pendingMeta);
      return;
    }
    const m = pendingMeta;
    const bmp = await createImageBitmap(ev.data);
    if (live.width !== bmp.width || live.height !== bmp.height) {
      live.width = bmp.width;
      live.height = bmp.height;
    }
    lctx.drawImage(bmp, 0, 0);
    bmp.close();
    if (m) drawOverlays(lctx, live.width, live.height, m);
  };
  ws.onclose = () => {
    showBanner(quitting
      ? 'Server stopped. To start it again, double-click “Start Vision Lab.command” in the project folder (or run ./run.sh). This page reconnects on its own.'
      : 'Lost connection to the Vision Lab server. Retrying…');
    setTimeout(connect, 1500);
  };
  ws.onopen = () => {
    showBanner(null);
    api('/api/state').then((s) => { checkBoot(s.boot_id); setPrivacy(!!s.paused); }).catch(() => {});
  };
}

function showBanner(msg) {
  $('#banner').hidden = !msg;
  $('#banner').textContent = msg || '';
}

let lastSlowUpdate = 0;
let quitting = false;
function onMeta(m) {
  checkBoot(m.boot_id);
  meta = m;
  quitting = false;
  showBanner(m.error);
  setPrivacy(serverPrivacy(m));
  if (m.paused) {
    lctx.fillStyle = '#0B1220';
    lctx.fillRect(0, 0, live.width, live.height);
    lctx.fillStyle = '#94A3B8';
    lctx.font = `500 ${Math.round(live.width / 40)}px ${getComputedStyle(document.body).getPropertyValue('--sans')}`;
    lctx.textAlign = 'center';
    lctx.fillText('PRIVATE: camera, screen and mic are stopped. Click PRIVATE to start recording again.', live.width / 2, live.height / 2);
    lctx.textAlign = 'start';
  }
  if (m.depth && m.depth !== depthSrc) {
    depthSrc = m.depth;
    depthImg.src = `data:image/jpeg;base64,${m.depth}`;
  }
  $('#hud-res').textContent = m.w ? `${m.w}×${m.h} · ${m.fps.pipeline} fps analysed` : 'no signal';
  const people = (m.objects || []).filter((o) => o.cls === 'person' && o.posture && o.posture !== 'unknown');
  $('#hud-pose').hidden = !people.length;
  $('#hud-pose').textContent = 'Pose: ' + people.map((p) => `#${p.id} ${p.posture}`).join(', ');
  if (m.events_rev !== lastEventsRev) {
    lastEventsRev = m.events_rev;
    loadEvents();
  }
  const now = performance.now();
  if (now - lastSlowUpdate > 400) {
    lastSlowUpdate = now;
    renderCounts(m.counts || {});
    renderBudget(m.timing || {});
    renderScene(m.scene, m.scene_error);
    renderStages();
  }
}

function colorFor(cls) {
  const o = (meta?.objects || []).find((x) => x.cls === cls);
  return o ? o.color : '#E0E0E0';
}

function renderCounts(counts) {
  const keys = Object.keys(counts);
  $('#counts').innerHTML = keys.length
    ? keys.map((k) => `<span class="chip" style="background:${colorFor(k)}">${esc(k)} ×${counts[k]}</span>`).join('')
    : '<span class="muted">Nothing detected</span>';
}

const BUDGET_COLORS = { detect: '#0D9488', track: '#99F6E4', segment: '#0E2A4F', pose: '#2DD4BF', depth: '#F59E0B', encode: '#94A3B8' };
function renderBudget(t) {
  const total = Object.values(t).reduce((a, b) => a + b, 0);
  const budget = 1000 / 30;
  $('#budget-bar').innerHTML = Object.entries(t).map(([k, v]) =>
    `<div style="width:${Math.min(100, (v / Math.max(budget, total)) * 100)}%;background:${BUDGET_COLORS[k] || '#888'}" title="${k} ${v} ms"></div>`).join('');
  $('#budget-legend').innerHTML = Object.entries(t).map(([k, v]) =>
    `<span><span style="color:${BUDGET_COLORS[k] || '#888'}">■</span> ${k} ${v.toFixed(1)}</span>`).join('') +
    '<span>Scene model runs outside the frame loop</span>';
  const el = $('#budget-total');
  el.textContent = `${total.toFixed(1)} ms used · ${total <= budget ? 'fits' : 'over budget, frames are skipped'}`;
  el.style.color = total <= budget ? 'var(--ok)' : 'var(--accent)';
}

function renderScene(scene, err) {
  if (!state.settings?.stages.describe) {
    $('#scene-text').innerHTML = '<span class="muted">Describe is off. Turn it on in the pipeline list.</span>';
    $('#scene-meta').textContent = '';
    $('#scene-tags').innerHTML = '';
    return;
  }
  if (err && !scene) { $('#scene-text').textContent = err; return; }
  if (!scene) return;
  $('#scene-text').textContent = scene.description;
  const ago = Math.max(0, Math.round(Date.now() / 1000 - scene.ts));
  $('#scene-meta').textContent = `${scene.model} · ${ago} s ago · took ${scene.latency} s`;
  const tags = [];
  if (scene.activity) tags.push(`activity: ${scene.activity}`);
  if (scene.people != null) tags.push(`people: ${scene.people}`);
  if (scene.pets != null) tags.push(`pets: ${scene.pets}`);
  $('#scene-tags').innerHTML = tags.map((t) => `<span class="tag">${esc(t)}</span>`).join('');
}

function drawOverlays(ctx, W, H, m) {
  const lw = Math.max(2, W / 520);
  const font = Math.round(Math.max(12, W / 85));
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (layers.depth && m.depth && depthImg.complete && depthImg.naturalWidth) {
    ctx.globalAlpha = 0.55;
    ctx.drawImage(depthImg, 0, 0, W, H);
    ctx.globalAlpha = 1;
  }
  if (layers.zones && !m.noZones) drawZones(ctx, W, H, state.zones, m.objects || [], false);
  const objs = m.objects || [];
  if (layers.masks) {
    for (const o of objs) {
      if (!o.poly) continue;
      ctx.beginPath();
      o.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      ctx.closePath();
      ctx.fillStyle = o.color + '66';
      ctx.fill();
    }
  }
  if (layers.tracks) {
    for (const o of objs) {
      if (!o.trail || o.trail.length < 2) continue;
      ctx.setLineDash([2, 8]);
      ctx.strokeStyle = o.color;
      ctx.lineWidth = lw * 1.2;
      ctx.beginPath();
      o.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  if (layers.boxes) {
    ctx.font = `500 ${font}px ${getComputedStyle(document.body).getPropertyValue('--mono')}`;
    for (const o of objs) {
      const [x1, y1, x2, y2] = [o.box[0] * W, o.box[1] * H, o.box[2] * W, o.box[3] * H];
      ctx.strokeStyle = o.color;
      ctx.lineWidth = lw;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      let label = `#${o.id} ${o.cls} ${o.conf.toFixed(2)}`;
      const dwell = Object.entries(o.dwell || {})[0];
      if (dwell) label += ` · ${dwell[0]} ${Math.floor(dwell[1] / 60)}:${String(dwell[1] % 60).padStart(2, '0')}`;
      const tw = ctx.measureText(label).width + 12;
      const ty = y1 - font - 8 >= 0 ? y1 - font - 8 : y1;
      ctx.fillStyle = o.color;
      ctx.fillRect(x1, ty, tw, font + 8);
      ctx.fillStyle = '#0D1117';
      ctx.fillText(label, x1 + 6, ty + font + 1);
    }
  }
  if (layers.pose) {
    for (const o of objs) {
      if (!o.kp) continue;
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = lw;
      for (const [a, b] of SKELETON) {
        const p = o.kp[a], q = o.kp[b];
        if (p[2] < 0.4 || q[2] < 0.4) continue;
        ctx.beginPath();
        ctx.moveTo(p[0] * W, p[1] * H);
        ctx.lineTo(q[0] * W, q[1] * H);
        ctx.stroke();
      }
      ctx.fillStyle = '#F5A524';
      for (const p of o.kp) {
        if (p[2] < 0.4) continue;
        ctx.beginPath();
        ctx.arc(p[0] * W, p[1] * H, lw * 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

function drawZones(ctx, W, H, zones, objs, handles, selectedId) {
  const lw = Math.max(2, W / 520);
  const font = Math.round(Math.max(12, W / 85));
  ctx.font = `500 ${font}px ${getComputedStyle(document.body).getPropertyValue('--mono')}`;
  zones.forEach((z) => {
    const pts = z.points.map(([x, y]) => [x * W, y * H]);
    ctx.strokeStyle = z.color;
    ctx.lineWidth = z.id === selectedId ? lw * 1.6 : lw;
    if (z.kind === 'zone') {
      ctx.setLineDash(handles ? [] : [8, 6]);
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = z.color + (handles ? '22' : '10');
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      const [[x1, y1], [x2, y2]] = pts;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      // arrow toward the "in" side
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      const len = Math.hypot(x2 - x1, y2 - y1) || 1;
      const nx = -(y2 - y1) / len, ny = (x2 - x1) / len;
      const ax = mx + nx * 30, ay = my + ny * 30;
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(ax, ay);
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - nx * 10 + ny * 7, ay - ny * 10 - nx * 7);
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - nx * 10 - ny * 7, ay - ny * 10 + nx * 7);
      ctx.stroke();
    }
    let label = z.name + (z.kind === 'line' ? ' (arrow = in)' : '');
    if (!handles && z.kind === 'zone') {
      const n = objs.filter((o) => (o.zones || []).includes(z.name)).length;
      if (n) label += ` · ${n} inside`;
    }
    const minX = Math.min(...pts.map((p) => p[0])), minY = Math.min(...pts.map((p) => p[1]));
    const tw = ctx.measureText(label).width + 12;
    const ly = Math.max(0, minY - font - 10);
    ctx.fillStyle = z.color;
    ctx.fillRect(minX, ly, tw, font + 8);
    ctx.fillStyle = '#0D1117';
    ctx.fillText(label, minX + 6, ly + font + 1);
    if (handles) {
      ctx.fillStyle = z.color;
      ctx.strokeStyle = '#0D1117';
      for (const [x, y] of pts) {
        ctx.fillRect(x - 6, y - 6, 12, 12);
        ctx.strokeRect(x - 6, y - 6, 12, 12);
      }
    }
  });
}

// ---------------- events & ask ----------------
async function loadEvents() {
  const evs = await api('/api/events?limit=40').catch(() => []);
  $('#events').innerHTML = evs.length ? evs.map((e) => `
    <div class="event"><span class="t">${fmtTime(e.ts)}</span><span class="dot" style="background:${esc(e.color)}"></span>
      <div style="min-width:0"><div>${esc(e.text)}</div>
      ${e.vlm_text ? `<div class="vlm">“${esc(e.vlm_text)}”</div>` : ''}
      ${e.keyframe_id ? `<a href="/api/keyframe/${e.keyframe_id}.jpg" target="_blank" rel="noopener">View frame</a>` : ''}</div></div>`).join('')
    : '<div class="muted small">No events yet. Things that come into view, zone entries and rule hits appear here.</div>';
}

$('#ask-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('#ask').value.trim();
  if (!q) return;
  const out = $('#ask-answer');
  out.hidden = false;
  out.textContent = 'Thinking…';
  try {
    const r = await api('/api/ask', { method: 'POST', body: { question: q } });
    out.textContent = `${r.answer}\n\n— ${r.model}, ${r.latency} s`;
  } catch (err) { out.textContent = err.message; }
});

// Snapshot and Record video buttons were removed (feat/voice-log); /api/snapshot and /api/record stay server-side, unused.

const PRIVACY_TIMEOUT_MS = 8000;
// the button only ever renders server-confirmed state (POST/GET /api/privacy, /api/state, ws meta).
let privacyPending = false;
const serverPrivacy = (m) => !!(m.capture_paused ?? m.privacy_mode ?? m.paused);
function renderPrivacy(state) {     // state: true (PRIVATE), false (RECORDING) or 'pending'
  const b = $('#power');
  const pending = state === 'pending';
  b.classList.toggle('pending', pending);
  b.classList.toggle('on', state === true);
  b.classList.toggle('off', state === false);
  b.disabled = pending;
  b.setAttribute('aria-pressed', String(state === true));
  $('#privacy-label').textContent = pending ? 'SWITCHING…' : state ? 'PRIVATE' : 'RECORDING';
}
let lastPrivacy = null;
function privacyChanged(on) {        // keep the header mic button in step with Privacy mode
  if (lastPrivacy === on) return;
  lastPrivacy = on;
  if (window.vlabRefreshMic) window.vlabRefreshMic();
}
function setPrivacy(on) {           // from server truth only; ignored while a request is in flight
  if (!privacyPending) renderPrivacy(!!on);
  if (!privacyPending) privacyChanged(!!on);
}
async function syncPrivacy() {
  // Re-read the server's true state after a failed request.
  try {
    const s = await api('/api/privacy', { signal: AbortSignal.timeout(PRIVACY_TIMEOUT_MS) });
    if (meta) meta.paused = !!s.privacy_mode;
    renderPrivacy(!!s.privacy_mode);
    privacyChanged(!!s.privacy_mode);
  } catch {
    renderPrivacy('pending');       // unknown: stay neutral until ws meta / reconnect reports the truth
    $('#power').disabled = false;
  }
}
$('#power').addEventListener('click', async () => {
  if (privacyPending) return;
  const turnOn = !(meta ? serverPrivacy(meta) : false);
  const want = turnOn ? 'on' : 'off';
  privacyPending = true;
  renderPrivacy('pending');
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), PRIVACY_TIMEOUT_MS);
  try {
    const r = await api('/api/privacy', { method: 'POST', body: { on: turnOn }, signal: ctl.signal });
    privacyPending = false;
    if (meta) meta.paused = !!r.paused;
    renderPrivacy(!!r.paused);
    privacyChanged(!!r.paused);
    if (!!r.paused !== turnOn) {
      showBanner(`Privacy mode did not turn ${want}: the server reports ${r.paused ? 'PRIVATE' : 'RECORDING'}.`);
    } else if (turnOn && !r.released) {
      showBanner('Privacy mode is on (PRIVATE), but the camera has not been released yet. The camera light may still be on.');
    } else {
      showBanner(null);
    }
  } catch (e) {
    privacyPending = false;
    const why = e.name === 'AbortError' ? `no response after ${PRIVACY_TIMEOUT_MS / 1000}s` : e.message;
    showBanner(`Could not turn Privacy mode ${want}: ${why}.`);
    await syncPrivacy();
  } finally { clearTimeout(timer); }
});
$('#quit').addEventListener('click', async () => {
  if (!confirm('Shut down the Vision Lab server? Any recording in progress is saved first.')) return;
  quitting = true;
  await api('/api/shutdown', { method: 'POST' }).catch(() => {});
});

let toastTimer;
function showToast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

// ---------------- zones editor ----------------
const zc = $('#zcanvas');
const zctx = zc.getContext('2d');
const pausedImg = new Image();
let tool = 'select';
let draft = [];          // points being drawn
let candidate = null;    // finished shape waiting for a name: {kind, points}
let drag = null;         // {zone, idx}
let selectedZone = null;

const HINTS = {
  select: 'Drag a corner to reshape a zone. Zones save automatically.',
  polygon: 'Click to add corners. Click the first corner (or press Enter) to finish.',
  line: 'Click two points. People or pets crossing the line toward the arrow count as "in".',
  sam: 'Click any object. SAM outlines it, and you can save the outline as a zone.',
};
function setTool(t) {
  tool = t;
  draft = [];
  document.querySelectorAll('.tool').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  $('#tool-hint').textContent = HINTS[t];
  drawZoneEditor();
}
document.querySelectorAll('.tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));

function loadPausedFrame() {
  pausedImg.onload = () => {
    zc.width = pausedImg.naturalWidth;
    zc.height = pausedImg.naturalHeight;
    $('#paused-at').textContent = `Frame paused at ${VLTime.clock(Date.now() / 1000)}`;
    drawZoneEditor();
  };
  pausedImg.onerror = () => { $('#paused-at').textContent = 'No camera frame available'; };
  pausedImg.src = `/api/frame.jpg?pause=1&t=${Date.now()}`;
}
$('#refresh-frame').addEventListener('click', loadPausedFrame);

function drawZoneEditor() {
  const W = zc.width, H = zc.height;
  zctx.clearRect(0, 0, W, H);
  if (pausedImg.complete && pausedImg.naturalWidth) zctx.drawImage(pausedImg, 0, 0, W, H);
  drawZones(zctx, W, H, state.zones, [], true, selectedZone);
  const pts = candidate ? candidate.points : draft;
  if (pts.length) {
    zctx.strokeStyle = '#FF8CC0';
    zctx.fillStyle = '#FF8CC055';
    zctx.lineWidth = 3;
    zctx.beginPath();
    pts.forEach(([x, y], i) => (i ? zctx.lineTo(x * W, y * H) : zctx.moveTo(x * W, y * H)));
    if (candidate && candidate.kind === 'zone') { zctx.closePath(); zctx.fill(); }
    zctx.stroke();
    for (const [x, y] of pts) { zctx.fillStyle = '#FF8CC0'; zctx.fillRect(x * W - 5, y * H - 5, 10, 10); }
  }
}

function canvasPoint(ev) {
  const r = zc.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
}
function nearPx(a, b, px = 14) {
  const r = zc.getBoundingClientRect();
  return Math.hypot((a[0] - b[0]) * r.width, (a[1] - b[1]) * r.height) < px;
}

function finishShape(kind, points) {
  candidate = { kind, points };
  draft = [];
  $('#name-form').hidden = false;
  $('#zone-name').value = kind === 'line' ? 'Doorway line' : `Zone ${state.zones.filter((z) => z.kind === 'zone').length + 1}`;
  $('#zone-name').select();
  drawZoneEditor();
}

zc.addEventListener('mousedown', async (ev) => {
  const p = canvasPoint(ev);
  if (candidate) return;
  if (tool === 'select') {
    for (const z of state.zones) {
      const idx = z.points.findIndex((q) => nearPx(p, q));
      if (idx >= 0) { drag = { zone: z, idx }; selectedZone = z.id; return; }
    }
    return;
  }
  if (tool === 'polygon') {
    if (draft.length >= 3 && nearPx(p, draft[0])) return finishShape('zone', draft);
    draft.push(p);
  } else if (tool === 'line') {
    draft.push(p);
    if (draft.length === 2) return finishShape('line', draft);
  } else if (tool === 'sam') {
    $('#tool-hint').textContent = 'Segmenting… (the first use downloads SAM 2.1, about 160 MB)';
    try {
      const r = await api('/api/segment', { method: 'POST', body: { x: p[0], y: p[1] } });
      finishShape('zone', r.points);
      $('#zone-name').value = 'Object zone';
      $('#tool-hint').textContent = HINTS.sam;
    } catch (e) { $('#tool-hint').textContent = e.message; }
  }
  drawZoneEditor();
});
zc.addEventListener('mousemove', (ev) => {
  if (!drag) return;
  const [x, y] = canvasPoint(ev);
  drag.zone.points[drag.idx] = [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  drawZoneEditor();
});
window.addEventListener('mouseup', () => {
  if (drag) { drag = null; saveZones(); }
});
window.addEventListener('keydown', (ev) => {
  if ($('#view-zones').hidden) return;
  if (ev.key === 'Enter' && tool === 'polygon' && draft.length >= 3 && !candidate) finishShape('zone', draft);
  if (ev.key === 'Escape') { draft = []; cancelName(); }
});

$('#name-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const id = 'z' + Date.now().toString(36);
  const color = ZONE_COLORS[state.zones.length % ZONE_COLORS.length];
  state.zones.push({ id, name: $('#zone-name').value.trim() || 'Zone', kind: candidate.kind, points: candidate.points, color });
  candidate = null;
  $('#name-form').hidden = true;
  saveZones();
});
function cancelName() { candidate = null; $('#name-form').hidden = true; drawZoneEditor(); }
$('#name-cancel').addEventListener('click', cancelName);

async function saveZones() {
  state.zones = await api('/api/zones', { method: 'PUT', body: state.zones });
  renderZoneList();
  renderRules();
  drawZoneEditor();
}

function renderZoneList() {
  $('#zone-list').innerHTML = state.zones.length ? state.zones.map((z) => `
    <div class="zone-row"><span class="dot" style="background:${z.color}"></span>
      <label class="sr-only" for="zn-${z.id}">Zone name</label>
      <input id="zn-${z.id}" type="text" value="${esc(z.name)}" data-rename="${z.id}">
      <span class="muted small">${z.kind === 'line' ? 'line' : `${z.points.length} corners`}</span>
      <button type="button" class="btn danger" data-delzone="${z.id}">Delete</button></div>`).join('')
    : '<div class="muted small">No zones yet. Pick a drawing tool above and draw on the frame.</div>';
}
$('#zone-list').addEventListener('change', (e) => {
  const id = e.target.dataset.rename;
  if (!id) return;
  state.zones.find((z) => z.id === id).name = e.target.value.trim() || 'Zone';
  saveZones();
});
$('#zone-list').addEventListener('click', (e) => {
  const id = e.target.dataset.delzone;
  if (!id) return;
  state.zones = state.zones.filter((z) => z.id !== id);
  saveZones();
});

// ---------------- rules ----------------
const EVENTS = {
  appears: 'comes into view',
  enters_zone: 'enters a zone',
  leaves_zone: 'leaves a zone',
  in_zone_for: 'stays in a zone',
  crosses_line: 'crosses a line',
  posture_change: 'changes posture',
};
let editing = null;

function ruleSummary(r) {
  const target = state.zones.find((z) => z.id === r.target);
  const what = `a <b>${esc(r.object === 'any' ? 'tracked object' : r.object)}</b>`;
  let when = `${what} ${EVENTS[r.event]}`;
  if (target) when = `${what} ${EVENTS[r.event].replace(/a (zone|line)/, `<b>${esc(target.name)}</b>`)}`;
  const cond = [];
  if (r.event === 'in_zone_for') cond.push(`more than ${Math.round(r.duration_s / 60)} min`);
  if (r.posture && r.posture !== 'any') cond.push(`posture = ${r.posture}`);
  if (r.event === 'crosses_line' && r.direction !== 'any') cond.push(`direction ${r.direction}`);
  cond.push(`at most every ${r.cooldown_s} s`);
  const then = ['log event'];
  if (r.actions.snapshot) then.push('save frame');
  if (r.actions.notify) then.push('macOS notification');
  if (r.actions.vlm_prompt) then.push(`ask: “${esc(r.actions.vlm_prompt)}”`);
  return `<div class="grid"><span>When</span><span>${when}</span><span>If</span><span>${cond.join(', ')}</span><span>Then</span><span>${then.join(' · ')}</span></div>`;
}

function renderRules() {
  $('#rules-title').textContent = `Rules · ${state.rules.filter((r) => r.enabled).length} active`;
  $('#rules').innerHTML = state.rules.map((r) => `
    <article class="card pad rule">
      <div class="row between"><h3>${esc(r.name)}</h3>
        <label class="check small"><input type="checkbox" data-toggle="${r.id}" ${r.enabled ? 'checked' : ''}> On</label></div>
      ${ruleSummary(r)}
      <div class="row" style="justify-content:flex-end">
        <button type="button" class="btn danger" data-delrule="${r.id}">Delete</button>
        <button type="button" class="btn" data-editrule="${r.id}">Edit</button></div>
    </article>`).join('');
  renderRuleEditor();
}

function renderRuleEditor() {
  const r = editing;
  if (!r) { $('#rule-editor').innerHTML = ''; return; }
  const zoneOpts = (kind) => state.zones.filter((z) => z.kind === kind)
    .map((z) => `<option value="${z.id}" ${z.id === r.target ? 'selected' : ''}>${esc(z.name)}</option>`).join('');
  const needsZone = ['enters_zone', 'leaves_zone', 'in_zone_for'].includes(r.event);
  const needsLine = r.event === 'crosses_line';
  const opts = (list, cur) => list.map((v) => `<option ${v === cur ? 'selected' : ''}>${esc(v)}</option>`).join('');
  $('#rule-editor').innerHTML = `
  <article class="card pad rule editing stack">
    <label class="field">Rule name <input type="text" id="re-name" value="${esc(r.name)}"></label>
    <div class="fields">
      <label class="field">Object <select id="re-object">${opts(['any', ...state.classes], r.object)}</select></label>
      <label class="field">Event <select id="re-event">${Object.entries(EVENTS).map(([k, v]) => `<option value="${k}" ${k === r.event ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
    </div>
    <div class="fields">
      ${needsZone ? `<label class="field">Zone <select id="re-target">${zoneOpts('zone') || '<option value="">Draw a zone first</option>'}</select></label>` : ''}
      ${needsLine ? `<label class="field">Line <select id="re-target">${zoneOpts('line') || '<option value="">Draw a line first</option>'}</select></label>
        <label class="field">Direction <select id="re-direction">${opts(['any', 'in', 'out'], r.direction)}</select></label>` : ''}
      ${r.event === 'in_zone_for' ? `<label class="field">For more than (min) <input type="number" id="re-duration" min="0" step="0.5" value="${r.duration_s / 60}"></label>` : ''}
      ${['in_zone_for', 'posture_change'].includes(r.event) ? `<label class="field">Posture <select id="re-posture">${opts(['any', 'standing', 'sitting', 'lying'], r.posture)}</select></label>` : ''}
      <label class="field">Cooldown (s) <input type="number" id="re-cooldown" min="0" value="${r.cooldown_s}"></label>
    </div>
    <label class="field">Then ask the vision model (optional)
      <textarea id="re-prompt" rows="3">${esc(r.actions.vlm_prompt)}</textarea></label>
    <label class="check"><input type="checkbox" id="re-snapshot" ${r.actions.snapshot ? 'checked' : ''}> Save the frame with the event</label>
    <label class="check"><input type="checkbox" id="re-notify" ${r.actions.notify ? 'checked' : ''}> Show a macOS notification</label>
    <div class="answer" id="re-answer" hidden></div>
    <div class="row wrap" style="justify-content:flex-end">
      <button type="button" class="btn" id="re-cancel">Cancel</button>
      <button type="button" class="btn" id="re-test">Test on paused frame</button>
      <button type="button" class="btn primary" id="re-save">Save rule</button>
    </div>
  </article>`;
  const read = () => {
    const v = (id) => $(id)?.value;
    editing = {
      ...editing,
      name: v('#re-name').trim() || 'Untitled rule',
      object: v('#re-object'),
      event: v('#re-event'),
      target: v('#re-target') || null,
      direction: v('#re-direction') || 'any',
      duration_s: $('#re-duration') ? Number(v('#re-duration')) * 60 : editing.duration_s,
      posture: v('#re-posture') || 'any',
      cooldown_s: Number(v('#re-cooldown')) || 0,
      actions: { vlm_prompt: v('#re-prompt').trim(), snapshot: $('#re-snapshot').checked, notify: $('#re-notify').checked },
    };
  };
  $('#re-event').addEventListener('change', () => { read(); editing.target = null; renderRuleEditor(); });
  $('#re-cancel').addEventListener('click', () => { editing = null; renderRuleEditor(); });
  $('#re-test').addEventListener('click', async () => {
    read();
    const out = $('#re-answer');
    out.hidden = false;
    if (!editing.actions.vlm_prompt) { out.textContent = 'Add a question for the vision model first.'; return; }
    out.textContent = 'Asking the vision model…';
    if (!pausedImg.naturalWidth) loadPausedFrame();
    try {
      const r2 = await api('/api/rule_test', { method: 'POST', body: { prompt: editing.actions.vlm_prompt } });
      out.textContent = `${r2.answer}\n\n— ${r2.model}, ${r2.latency} s`;
    } catch (e) { out.textContent = e.message; }
  });
  $('#re-save').addEventListener('click', async () => {
    read();
    const i = state.rules.findIndex((x) => x.id === editing.id);
    if (i >= 0) state.rules[i] = editing; else state.rules.push(editing);
    editing = null;
    await saveRules();
  });
}

async function saveRules() {
  state.rules = await api('/api/rules', { method: 'PUT', body: state.rules });
  renderRules();
}
$('#new-rule').addEventListener('click', () => {
  editing = { id: 'r' + Date.now().toString(36), name: 'New rule', enabled: true, object: 'person', event: 'enters_zone',
    target: null, duration_s: 0, posture: 'any', direction: 'any', cooldown_s: 60,
    actions: { vlm_prompt: '', snapshot: true, notify: false } };
  renderRuleEditor();
});
$('#rules').addEventListener('click', (e) => {
  const { editrule, delrule } = e.target.dataset;
  if (editrule) { editing = structuredClone(state.rules.find((r) => r.id === editrule)); renderRuleEditor(); }
  if (delrule) { state.rules = state.rules.filter((r) => r.id !== delrule); saveRules(); }
});
$('#rules').addEventListener('change', (e) => {
  const id = e.target.dataset.toggle;
  if (!id) return;
  state.rules.find((r) => r.id === id).enabled = e.target.checked;
  saveRules();
});

// ---------------- timeline & search ----------------
$('#search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('#q').value.trim();
  if (!q) return;
  $('#search-meta').textContent = 'Searching…';
  try {
    const r = await api(`/api/search?q=${encodeURIComponent(q)}&range=${$('#range').value}`);
    const how = r.method === 'siglip' ? 'matched with SigLIP image embeddings' : 'matched against captions (image search unavailable)';
    $('#search-meta').textContent = `${r.results.length} matches out of ${r.total} keyframes, ${how}.`;
    $('#shots').innerHTML = r.results.map((s) => `
      <figure class="shot"><a href="/api/keyframe/${s.id}.jpg" target="_blank" rel="noopener"><img src="/api/keyframe/${s.id}.jpg" alt="${esc(s.caption || 'Keyframe at ' + fmtTime(s.ts))}" loading="lazy"></a>
        <figcaption><div class="row between mono tiny"><span>${VLTime.dayTime(s.ts)}</span>
        <span style="color:#0F766E">${s.score != null ? s.score.toFixed(3) : ''}</span></div>
        <div class="muted small">${esc(s.caption || '')}</div></figcaption></figure>`).join('');
  } catch (err) { $('#search-meta').textContent = err.message; }
});

// One shared scale for the Captures and Object activity boxes: same window, same zoom, synced scroll.
const TL = { start: null, zoom: 1 };
const dayStart = (d) => { const x = new Date(d); x.setHours(0, 0, 0, 0); return x.getTime() / 1000; };
const tlEnd = () => { const d = new Date(TL.start * 1000); d.setDate(d.getDate() + 1); return d.getTime() / 1000; };
const tlPct = (ts) => ((ts - TL.start) / (tlEnd() - TL.start)) * 100;
function tlAxis() {
  const step = TL.zoom >= 24 ? 900 : TL.zoom >= 8 ? 1800 : TL.zoom >= 4 ? 3600 : TL.zoom >= 2 ? 3 * 3600 : 6 * 3600;
  const ticks = [];
  for (let t = TL.start; t <= tlEnd() + 1; t += step) {
    const lab = VLTime.tick(t);   // "6 AM", "10:30 AM"; end of day reads 12 AM
    ticks.push(`<span style="left:${tlPct(t)}%">${lab}</span>`);
  }
  return `<div class="lane"><span></span><div class="axis tl-axis">${ticks.join('')}</div></div>`;
}
const bar = (a, b, color, title) => `<div style="left:${tlPct(a)}%;width:${Math.max(0.08, tlPct(b) - tlPct(a))}%;background:${color}" title="${title}"></div>`;
const dur = (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`);
const CAP_COLORS = { video: '#0D9488', audio: '#2DD4BF', screen: '#0E2A4F' };
function renderCaptures(c) {
  const privacy = c.privacy.map(([a, b]) => bar(a, b, 'repeating-linear-gradient(45deg,#CBD5E1 0 4px,transparent 4px 8px)', `Privacy mode ${fmtTime(a)}–${fmtTime(b)}`)).join('');
  const rows = c.rows.map((r) => `
    <div class="lane"><span><span style="color:${CAP_COLORS[r.key]}">●</span> ${esc(r.label)} <span class="muted tiny">${dur(r.seconds)}</span></span>
      <div class="track">${privacy}${r.bars.map(([a, b]) => bar(a, b, CAP_COLORS[r.key], `${esc(r.label)} ${fmtTime(a)}–${fmtTime(b)}`)).join('')}</div></div>`).join('');
  $('#cap-lanes').innerHTML = rows + tlAxis();
  $('#cap-meta').textContent = c.privacy.length ? 'Hatched = Privacy mode' : '';
}
function tlApplyZoom() {
  document.querySelectorAll('.tl-scroll > .lanes').forEach((l) => (l.style.width = `${TL.zoom * 100}%`));
}
async function loadTimeline() {
  if (TL.start == null) TL.start = dayStart(Date.now());
  const d = new Date(TL.start * 1000);
  $('#tl-day').value = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  $('#tl-day-label').textContent = d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
  const q = `start=${TL.start}&end=${tlEnd()}`;
  const [t, c] = await Promise.all([api(`/api/timeline?${q}`).catch(() => null), api(`/api/timeline/captures?${q}`).catch(() => null)]);
  if (c) renderCaptures(c);
  if (t) {
    const lanes = t.lanes.map((l) => `
    <div class="lane"><span><span style="color:${l.color}">●</span> ${esc(l.cls)} <span class="muted tiny">${Math.round(l.seconds / 60)} min</span></span>
      <div class="track">${l.spans.map(([a, b]) => bar(a, b, l.color, `${fmtTime(a)}–${fmtTime(b)}`)).join('')}</div></div>`).join('');
    const rules = `<div class="lane"><span><span style="color:#0E2A4F">●</span> Rule fired</span>
      <div class="track">${t.rules.map((ts) => `<div style="left:${tlPct(ts)}%;width:3px;background:#0E2A4F" title="${fmtTime(ts)}"></div>`).join('')}</div></div>`;
    $('#lanes').innerHTML = (lanes || '<div class="muted small">Nothing was in view long enough on this day.</div>') + rules + tlAxis();
  }
  tlApplyZoom();
}
let tlSyncing = false;
document.querySelectorAll('.tl-scroll').forEach((el) => el.addEventListener('scroll', () => {
  if (tlSyncing) return; tlSyncing = true;
  document.querySelectorAll('.tl-scroll').forEach((o) => { if (o !== el) o.scrollLeft = el.scrollLeft; });
  requestAnimationFrame(() => (tlSyncing = false));
}));
const tlShift = (days) => { const d = new Date(TL.start * 1000); d.setDate(d.getDate() + days); TL.start = d.getTime() / 1000; loadTimeline(); };
$('#tl-prev').addEventListener('click', () => tlShift(-1));
$('#tl-next').addEventListener('click', () => tlShift(1));
$('#tl-today').addEventListener('click', () => { TL.start = dayStart(Date.now()); loadTimeline(); });
$('#tl-day').addEventListener('change', (e) => { if (e.target.value) { TL.start = dayStart(`${e.target.value}T00:00:00`); loadTimeline(); } });
$('#tl-zoom').addEventListener('change', async (e) => {   // keep the centre of the view in place; re-render ticks
  const sc = document.querySelector('.tl-scroll'), frac = sc ? (sc.scrollLeft + sc.clientWidth / 2) / sc.scrollWidth : 0;
  TL.zoom = Number(e.target.value);
  await loadTimeline();
  document.querySelectorAll('.tl-scroll').forEach((o) => (o.scrollLeft = frac * o.scrollWidth - o.clientWidth / 2));
});
$('#tl-refresh').addEventListener('click', loadTimeline);

async function summarize(question) {
  const btn = $('#summarize');
  btn.disabled = true;
  $('#summary-text').textContent = question ? 'Thinking…' : 'Writing a summary…';
  try {
    const r = await api('/api/summary', { method: 'POST', body: question ? { question } : {} });
    $('#summary-text').textContent = r.text;
    $('#summary-meta').textContent = `Written by ${r.model} from ${r.events} logged events. No video leaves this Mac.`;
  } catch (e) { $('#summary-text').textContent = e.message; }
  btn.disabled = false;
}
$('#summarize').addEventListener('click', () => summarize());
$('#day-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = $('#ask-day').value.trim();
  if (q) summarize(q);
});

// ---------------- model lab ----------------
const GB = 1024 ** 3;
async function loadMemory() {
  const m = await api('/api/memory').catch(() => null);
  if (!m) return;
  const parts = [{ name: 'Vision Lab', bytes: m.lab_process + m.lab_gpu, color: '#0D9488' },
    ...m.ollama.map((o, i) => ({ name: o.name, bytes: o.bytes, color: ['#0E2A4F', '#2DD4BF', '#64748B'][i % 3] }))];
  const known = parts.reduce((a, p) => a + p.bytes, 0);
  parts.push({ name: 'Everything else', bytes: Math.max(0, m.used - known), color: '#CBD5E1' });
  $('#mem-total').textContent = `${(m.used / GB).toFixed(1)} GB of ${(m.total / GB).toFixed(0)} GB in use`;
  $('#mem-bar').innerHTML = parts.map((p) => `<div style="width:${(p.bytes / m.total) * 100}%;background:${p.color}"></div>`).join('');
  $('#mem-legend').innerHTML = parts.map((p) => `<span><span style="color:${p.color}">■</span> ${esc(p.name)} ${(p.bytes / GB).toFixed(1)} GB</span>`).join('') +
    `<span>${((m.total - m.used) / GB).toFixed(0)} GB free for bigger models</span>`;
}

function speedBar(b) {
  if (!b) return '<span class="muted tiny">not measured</span>';
  const v = b.fps ?? 1000;
  const w = Math.max(3, Math.min(100, Math.round((Math.log10(v * 10 + 1) / Math.log10(3001)) * 100)));
  const c = v >= 30 ? '#0D9488' : v >= 10 ? '#2DD4BF' : '#F59E0B';
  return `<div class="speed"><div class="meter"><div style="width:${w}%;background:${c}"></div></div><span class="mono tiny" style="width:64px;text-align:right">${esc(b.label)}</span></div>`;
}

function fmtDay(d) {
  if (!d || d === 'now' || d === 'just now') return d || '';
  const t = new Date(`${d}T12:00:00`);
  return isNaN(t) ? d : t.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function benchTip(b) {
  // Provenance lives in the hover tooltip only - nothing is shown below the number.
  const parts = b.kind === 'published'
    ? [`published · ${b.source || ''}`, b.note, b.url]
    : [`measured here · ${fmtDay(b.date)}`, b.note, b.machine];
  return parts.filter(Boolean).join('\n');
}

function benchList(r) {
  const items = (r.benchmarks || []).map((b) =>
    `<li class="bm-${b.kind === 'published' ? 'pub' : 'here'}" title="${esc(benchTip(b))}"><span class="bm-label">${esc(b.label)}</span> <b class="bm-val">${esc(b.value)}</b></li>`);
  if (!(r.benchmarks || []).some((b) => b.kind === 'measured')) {
    items.push(`<li class="bm-none"${r.measure_note ? ` title="${esc(r.measure_note)}"` : ''}><span class="bm-label">Speed &amp; memory here</span> <span class="muted">not measured</span></li>`);
  }
  return `<ul class="bm">${items.join('')}</ul>`;
}

function usageCell(u) {
  if (!u) return '';
  const where = u.where.length ? `<ul class="where">${u.where.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : '';
  const loaded = u.loaded ? `<div class="loaded"><span class="dot ok"></span>Loaded now: ${esc(u.loaded_where.join(', '))}</div>` : '';
  return `<span class="use ${u.status}">${esc(u.label)}</span>${where}${loaded}`;
}

function actionCell(r) {
  if (r.pulling) return '<span class="muted small">Downloading…</span>';
  if (!r.installed) return r.kind === 'info' ? '<span class="muted small">Not installed</span>'
    : `<button type="button" class="btn primary" data-download="${esc(r.key)}">Download</button>`;
  return r.benchable ? `<button type="button" class="btn" data-bench="${esc(r.key)}">Benchmark</button>` : '';
}

async function loadModels() {
  loadMemory();
  const rows = await api('/api/models').catch(() => []);
  const n = (st) => rows.filter((r) => r.usage && r.usage.status === st).length;
  const sum = $('#catalog-summary');
  if (sum) sum.textContent = rows.length ? `${n('active')} in use · ${n('on_demand')} on demand · ${n('idle')} not in use · ${rows.filter((r) => r.usage && r.usage.loaded).length} loaded now` : '';
  let group = '';
  $('#catalog tbody').innerHTML = rows.map((r) => {
    const head = r.group && r.group !== group ? `<tr class="group"><th colspan="6" scope="colgroup">${esc(group = r.group)}</th></tr>` : '';
    return `${head}
    <tr class="use-${r.usage ? r.usage.status : 'idle'}"><td data-label="Model"><b>${esc(r.name)}</b>${r.bench ? `<div class="live-bench">${speedBar(r.bench)}</div>` : ''}</td>
      <td data-label="In use">${usageCell(r.usage)}</td>
      <td data-label="What it does"><div class="task">${esc(r.task)}</div><div class="gives">${esc(r.gives)}</div></td>
      <td data-label="Benchmarks">${benchList(r)}</td>
      <td data-label="Runs on"><span class="hw ${r.hw}">${r.hw}</span></td>
      <td data-label="">${actionCell(r)}</td></tr>`;
  }).join('');
  const dets = ['yolo11n', 'yolo11s', 'yolov8s-worldv2', 'yolo11s-seg'];
  const opt = (sel) => dets.map((d) => `<option ${d === sel ? 'selected' : ''}>${d}</option>`).join('');
  if (!$('#ab-a').options.length) {
    $('#ab-a').innerHTML = opt('yolo11n');
    $('#ab-b').innerHTML = opt('yolov8s-worldv2');
    $('#ab-classes').value = 'person, cup, laptop, phone, headphones, framed picture, plant';
  }
}
$('#catalog').addEventListener('click', async (e) => {
  const { bench, download } = e.target.dataset;
  if (!bench && !download) return;
  e.target.disabled = true;
  e.target.textContent = bench ? 'Running…' : 'Downloading…';
  try {
    await api(bench ? '/api/benchmark' : '/api/download', { method: 'POST', body: { key: bench || download } });
  } catch (err) { showBanner(err.message); }
  loadModels();
});

$('#ab-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  $('#ab').innerHTML = '<div class="muted">Running both models on the current frame…</div>';
  try {
    const r = await api('/api/compare', { method: 'POST', body: { a: $('#ab-a').value, b: $('#ab-b').value, classes: $('#ab-classes').value } });
    const img = new Image();
    img.src = `/api/paused.jpg?t=${Date.now()}`;
    await img.decode();
    $('#ab').innerHTML = r.results.map((x, i) => `
      <figure><canvas id="ab-${i}" width="${img.naturalWidth}" height="${img.naturalHeight}" aria-label="${esc(x.key)} detections"></canvas>
        <figcaption><b>${esc(x.key)}</b><span class="mono tiny muted">${x.error ? '' : `${x.objects.length} objects · ${x.ms} ms`}</span>
        <span class="muted small" style="flex-basis:100%">${x.error ? esc(x.error) : esc([...new Set(x.objects.map((o) => o.cls))].join(', ') || 'nothing found')}</span></figcaption></figure>`).join('');
    r.results.forEach((x, i) => {
      const c = $(`#ab-${i}`), ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      drawOverlays(ctx, c.width, c.height, { noZones: true, objects: x.objects.map((o, j) => ({ ...o, id: j + 1 })) });
    });
    loadModels();
  } catch (err) { $('#ab').innerHTML = `<div class="muted">${esc(err.message)}</div>`; }
});

// ---------------- boot ----------------
setTool('select');
loadState().then(() => { connect(); loadEvents(); });
setInterval(() => { if (!$('#view-models').hidden) loadMemory(); }, 5000);


// ---- Screens tab (local-only screenshots + fast-lane captions) ----
async function loadScreens() {
  const sel = document.getElementById('scr-date');
  const r = await fetch('/api/screens?limit=200' + (sel.value ? '&date=' + sel.value : '')).then((x) => x.json());
  const keep = sel.value;
  sel.innerHTML = '<option value="">Recent</option>' + r.days.map((d) => `<option value="${d}">${d}</option>`).join('');
  sel.value = keep;
  const grid = document.getElementById('scr-grid');
  grid.innerHTML = '';
  document.getElementById('scr-empty').hidden = r.items.length > 0;
  for (const it of r.items) {
    const fig = document.createElement('figure');
    fig.className = 'scr-item';
    const a = document.createElement('a');
    a.href = `/api/screens/${it.id}.jpg`; a.target = '_blank';
    const img = document.createElement('img');
    img.loading = 'lazy'; img.src = a.href; img.alt = it.caption || 'screen';
    img.onerror = () => { img.replaceWith(Object.assign(document.createElement('div'), { className: 'small', textContent: '(image pruned)' })); };
    a.appendChild(img);
    const cap = document.createElement('figcaption');
    const t = VLTime.dayTime(it.capture_ts);
    cap.innerHTML = '<div class="mono small"></div><div></div>';
    cap.firstChild.textContent = `${t}${it.label ? ' · ' + it.label : ''}${it.gemma_ms ? ' · ' + (it.gemma_ms / 1000).toFixed(1) + 's' : ''}`;
    cap.lastChild.textContent = it.status === 'ok' ? (it.caption || '').replace(/^\w+\s*-\s*/, '') : `(${it.status})`;
    fig.append(a, cap);
    grid.appendChild(fig);
  }
}
document.getElementById('scr-date').addEventListener('change', loadScreens);
document.getElementById('scr-refresh').addEventListener('click', loadScreens);
