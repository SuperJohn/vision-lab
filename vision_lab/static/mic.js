// Header mic toggle + the Microphone picker on the Voice tab, backed by /api/mic (persisted in the config table).
// The picker only changes the input device; it never turns the mic on (works during Privacy mode too).
// Privacy mode locks it: the button is disabled and reads "Mic off (private)" (the server also refuses with 409).
(() => {
  const $m = (s) => document.querySelector(s);
  const TITLE_ON = 'The microphone is on. Click to turn it off. Voice is only captured while recording mode is on.';
  const TITLE_OFF = 'The microphone is off. Click to turn it on.';
  const TITLE_PRIVATE = 'Privacy mode is on, so the microphone is off. Turn off Privacy mode (PRIVATE) to use it.';
  const render = (s) => {
    const b = $m('#mic-toggle'), priv = !!s.privacy, on = !!s.enabled && !priv;
    $m('#mic-dot').className = `dot ${on ? 'ok' : ''}`;
    $m('#mic-label').textContent = priv ? 'Mic off (private)' : on ? 'Mic on' : 'Mic off';
    b.classList.toggle('primary', on);
    b.classList.toggle('mic-locked', priv);
    b.disabled = priv;
    b.setAttribute('aria-disabled', String(priv));
    b.title = priv ? TITLE_PRIVATE : on ? TITLE_ON : TITLE_OFF;
    // Any browser mic stream must close while off/private. The old voice-chat UI that exposed vlabStopMic is gone
    // (feat/voice-log); the hook stays for whatever browser capture comes next.
    if (!on && window.vlabStopMic) window.vlabStopMic();
  };
  async function load() {
    const s = await fetch('/api/mic').then((r) => r.json());
    const sel = $m('#mic-device'); sel.innerHTML = '<option value="">System default</option>';
    s.devices.forEach((d) => { const o = document.createElement('option'); o.value = o.textContent = d.name;
      if (d.manufacturer && /hollyland/i.test(d.manufacturer)) o.textContent += ' (Hollyland Lark)'; sel.append(o); });
    sel.value = s.device || '';
    if (s.warning) console.warn(s.warning);
    render(s);
  }
  // Cheap re-read (no device scan); app.js calls this whenever Privacy mode changes.
  async function refresh() {
    try { render(await fetch('/api/mic/state').then((r) => r.json())); } catch (_) {}
  }
  window.vlabRefreshMic = refresh;
  $m('#mic-toggle').addEventListener('click', async () => {
    const r = await fetch('/api/mic/toggle', { method: 'POST' });
    if (r.status === 409) { const e = await r.json().catch(() => ({})); console.warn(e.detail); return refresh(); }
    render(await r.json());
  });
  $m('#mic-device').addEventListener('change', (e) => fetch('/api/mic', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: e.target.value || null }) }).then(load));
  load();
})();
