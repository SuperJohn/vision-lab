// Header mic toggle + input picker, backed by /api/mic (persisted in the config table).
(() => {
  const $m = (s) => document.querySelector(s);
  const render = (s) => {
    $m('#mic-dot').className = `dot ${s.enabled ? 'ok' : ''}`;
    $m('#mic-label').textContent = s.enabled ? 'Mic on' : 'Mic off';
    $m('#mic-toggle').classList.toggle('primary', !!s.enabled);
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
  $m('#mic-toggle').addEventListener('click', async () => {
    const s = await fetch('/api/mic/toggle', { method: 'POST' }).then((r) => r.json());
    if (!s.enabled && window.vlabStopMic) window.vlabStopMic();
    render(s);
  });
  $m('#mic-device').addEventListener('change', (e) => fetch('/api/mic', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: e.target.value || null }) }).then(load));
  load();
})();
