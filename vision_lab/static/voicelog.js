// Voice tab: a read-only log of captured voice as text (GET /api/voice/log), grouped by day, newest first.
// Today is expanded, older days collapsed; open/closed state is kept across refreshes.
(() => {
  const $l = (s) => document.querySelector(s);
  const DAYS = 7;
  const open = new Map();            // date -> bool (user's choice wins over the default)
  let pages = [], nextBefore = null, loading = false;

  const hms = (ts) => VLTime.clock(ts);          // "10:31:05 AM" (shared formatter, timefmt.js)
  const hm = (ts) => VLTime.clock(ts, false);    // "10:31 AM"
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const params = (before) => {
    const q = new URLSearchParams({ days: DAYS, replies: $l('#vl-replies').checked ? 1 : 0 });
    if (before) q.set('before', before);
    return q;
  };

  function row(e) {
    const r = el('div', `vl-row ${e.kind}`);
    r.append(el('span', 't', hms(e.ts)));
    const who = { speech: 'You', reply: 'Assistant', transcript: 'Voice' }[e.kind] || e.kind;
    r.append(el('span', 'who', who));
    r.append(el('span', 'txt grow', e.text));
    if (e.interrupted) r.append(el('span', 'flag', 'interrupted'));
    return r;
  }

  function day(d) {
    const det = el('details', 'vl-day');
    det.dataset.date = d.date;
    det.open = open.has(d.date) ? open.get(d.date) : !!d.today;
    const sum = el('summary');
    sum.append(el('span', 'vl-date', d.label));
    if (d.today) sum.append(el('span', 'vl-today', 'Today'));
    sum.append(el('span', 'muted small', `${d.count} ${d.count === 1 ? 'entry' : 'entries'}${d.replies ? ` · ${d.replies} replies` : ''}`));
    sum.append(el('span', 'grow'));
    sum.append(el('span', 'muted small mono', `${hm(d.first_ts)}–${hm(d.last_ts)}`));
    det.append(sum);
    const rows = el('div', 'vl-rows');
    d.entries.forEach((e) => rows.append(row(e)));
    det.append(rows);
    det.addEventListener('toggle', () => open.set(d.date, det.open));
    return det;
  }

  function render() {
    const days = pages.flatMap((p) => p.days);
    const box = $l('#vl-days'); box.replaceChildren(...days.map(day));
    $l('#vl-empty').hidden = days.length > 0;
    $l('#vl-more').hidden = !nextBefore;
    const n = days.reduce((a, d) => a + d.count, 0);
    $l('#vl-summary').textContent = days.length ? `${n} entries · ${days.length} day${days.length === 1 ? '' : 's'} loaded` : '';
  }

  async function fetchPage(before) {
    return fetch(`/api/voice/log?${params(before)}`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  }
  async function load() {                    // first page (or a full refresh)
    if (loading) return; loading = true;
    try { const p = await fetchPage(null); pages = [p]; nextBefore = p.next_before; render(); }
    catch (e) { $l('#vl-summary').textContent = `Could not load the voice log (${e.message})`; }
    finally { loading = false; }
  }
  async function more() {
    if (loading || !nextBefore) return; loading = true;
    try { const p = await fetchPage(nextBefore); pages.push(p); nextBefore = p.next_before; render(); }
    finally { loading = false; }
  }
  const setAll = (v) => document.querySelectorAll('#vl-days details').forEach((d) => { d.open = v; open.set(d.dataset.date, v); });

  $l('#vl-expand').addEventListener('click', () => setAll(true));
  $l('#vl-collapse').addEventListener('click', () => setAll(false));
  $l('#vl-refresh').addEventListener('click', load);
  $l('#vl-more').addEventListener('click', more);
  $l('#vl-replies').addEventListener('change', load);
  window.vlabLoadVoiceLog = load;            // app.js calls this when the Voice tab is shown
})();
