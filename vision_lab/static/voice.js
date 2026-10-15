// Voice tab: browser mic -> /ws/voice, plays streamed Kokoro audio, shows per-stage metrics.
(() => {
  const $v = (s) => document.querySelector(s);
  let ws = null, micCtx = null, stream = null, playCtx = null, playAt = 0, sources = [], pendingAudio = null, replyEl = null;

  const setState = (s) => { $v('#v-state').innerHTML = `<span class="dot ${s === 'off' ? '' : 'ok'}"></span>${s}`; };
  const add = (cls, text) => { const d = document.createElement('div'); d.className = `vmsg ${cls}`; d.textContent = text;
    $v('#v-log').append(d); $v('#v-log').scrollTop = 1e9; return d; };

  function stopPlayback() { sources.forEach((s) => { try { s.stop(); } catch (_) {} }); sources = []; playAt = 0; }

  function play(buf, sr) {
    if (!playCtx) playCtx = new AudioContext();
    const i16 = new Int16Array(buf), f = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
    const ab = playCtx.createBuffer(1, f.length, sr); ab.copyToChannel(f, 0);
    const src = playCtx.createBufferSource(); src.buffer = ab; src.connect(playCtx.destination);
    playAt = Math.max(playAt, playCtx.currentTime + 0.03); src.start(playAt); playAt += ab.duration;
    sources.push(src);
    src.onended = () => { sources = sources.filter((x) => x !== src);
      if (!sources.length && ws && ws.readyState === 1) { ws.send(JSON.stringify({ type: 'playback_done' })); setState(micCtx ? 'listening for you' : 'ready'); } };
  }

  function metrics(t) {
    const cell = (k, label) => `<div class="vmetric"><b>${t[k] ?? '–'}</b>${label}</div>`;
    $v('#v-metrics').innerHTML = cell('stt_ms', 'speech end → transcript') + cell('llm_first_token_ms', 'LLM first token')
      + cell('tts_first_audio_ms', 'first audio ready') + cell('tts_first_chunk_ms', 'TTS per first sentence')
      + cell('total_ms', 'whole reply generated') + `<div class="vmetric"><b>${window._vStats?.interruptions ?? 0}</b>interruptions</div>`;
    const tr = document.createElement('tr');
    const td = (x) => { const c = document.createElement('td'); c.textContent = x ?? '–'; return c; };
    tr.append(td(new Date(t.ts * 1000).toLocaleTimeString()), td(t.user), td(t.stt_ms), td(t.llm_first_token_ms),
      td(t.tts_first_audio_ms), td(t.total_ms), td(t.interrupted ? 'yes' : t.error ? 'error' : 'no'));
    $v('#v-turns tbody').prepend(tr);
  }

  function connect() {
    return new Promise((resolve) => {
      ws = new WebSocket(`ws://${location.host}/ws/voice`); ws.binaryType = 'arraybuffer';
      ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) { if (pendingAudio) play(e.data, pendingAudio.sr); pendingAudio = null; return; }
        const m = JSON.parse(e.data);
        if (m.type === 'ready') { window._vStats = m.stats; setState('ready'); resolve(); }
        else if (m.type === 'state') setState(m.state === 'listening' ? 'hearing you…' : m.state);
        else if (m.type === 'partial') $v('#v-partial').textContent = `… ${m.text}`;
        else if (m.type === 'transcript') { $v('#v-partial').textContent = ''; if (m.text) add('me', m.text); replyEl = null; }
        else if (m.type === 'token') { replyEl = replyEl || add('', ''); replyEl.textContent += m.text; }
        else if (m.type === 'audio') pendingAudio = m;
        else if (m.type === 'interrupt') { stopPlayback(); window._vStats.interruptions = m.interruptions; if (replyEl) replyEl.textContent += ' [interrupted]'; replyEl = null; }
        else if (m.type === 'metrics') { window._vStats = m.stats; metrics(m.turn); }
        else if (m.type === 'error') add('', `Error: ${m.error}`);
      };
      ws.onclose = () => { setState('off'); ws = null; };
      setState('loading models…');
    });
  }

  async function startMic() {
    if (!ws) await connect();
    const mic = await fetch('/api/mic').then((r) => r.json()).catch(() => ({ enabled: true }));
    if (!mic.enabled) throw new Error('microphone is turned off (header toggle)');
    const base = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
    stream = await navigator.mediaDevices.getUserMedia({ audio: base });
    if (mic.device) {
      const ins = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
      const hit = ins.find((d) => d.label === mic.device) || ins.find((d) => d.label.includes(mic.device));
      if (hit && stream.getAudioTracks()[0].getSettings().deviceId !== hit.deviceId) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await navigator.mediaDevices.getUserMedia({ audio: { ...base, deviceId: { exact: hit.deviceId } } });
      }
    }
    micCtx = new AudioContext();
    await micCtx.audioWorklet.addModule('/static/mic-worklet.js');
    const node = new AudioWorkletNode(micCtx, 'mic-16k');
    node.port.onmessage = (e) => { if (ws && ws.readyState === 1) ws.send(e.data.buffer); };
    micCtx.createMediaStreamSource(stream).connect(node);
    $v('#v-mic').textContent = 'Stop talking'; setState('listening for you');
  }
  function stopMic() {
    stream?.getTracks().forEach((t) => t.stop()); micCtx?.close(); stream = micCtx = null;
    $v('#v-mic').textContent = 'Start talking'; setState(ws ? 'ready' : 'off');
  }

  window.vlabStopMic = () => { if (micCtx) stopMic(); };
  $v('#v-mic').addEventListener('click', () => (micCtx ? stopMic() : startMic().catch((e) => add('', `Mic error: ${e.message}`))));
  $v('#v-stop').addEventListener('click', () => { stopPlayback(); ws?.send(JSON.stringify({ type: 'stop' })); });
  $v('#v-form').addEventListener('submit', async (e) => {
    e.preventDefault(); const text = $v('#v-text').value.trim(); if (!text) return;
    if (!ws) await connect(); stopPlayback(); add('me', text); replyEl = null; $v('#v-text').value = '';
    ws.send(JSON.stringify({ type: 'text', text }));
  });
  const post = (patch) => fetch('/api/voice/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) });
  $v('#v-voice').addEventListener('change', (e) => post({ voice: e.target.value }));
  $v('#v-engine').addEventListener('change', (e) => post({ tts_engine: e.target.value }));
  fetch('/api/voice/state').then((r) => r.json()).then((s) => { $v('#v-voice').value = s.settings.voice; s.events.filter((e) => e.kind === 'voice_turn').forEach(metrics); });
})();
