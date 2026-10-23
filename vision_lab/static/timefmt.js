// One shared clock formatter for the whole UI: 12-hour with AM/PM, seconds kept where shown.
//   VLTime.clock(ts)                 -> "10:31:05 AM"   (ts = epoch seconds, local time)
//   VLTime.clock(ts, false)          -> "10:31 AM"
//   VLTime.tick(ts)                  -> "10 AM" / "10:30 AM"  (axis labels: no seconds, drop ":00")
//   VLTime.dayTime(ts)               -> "Oct 20, 10:31:05 AM"
//   VLTime.range(a, b, seconds)      -> "10:31:05 AM–10:45:10 AM"
// Plain string building (no toLocale*), so every browser shows the same thing. Also loadable in node for tests.
(function (root) {
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const pad = (n) => String(n).padStart(2, '0');
  const parts = (ts) => {
    const d = ts instanceof Date ? ts : new Date(ts * 1000);
    const h = d.getHours();
    return { d, h12: h % 12 === 0 ? 12 : h % 12, ampm: h < 12 ? 'AM' : 'PM', m: d.getMinutes(), s: d.getSeconds() };
  };
  function clock(ts, seconds = true) {
    const p = parts(ts);
    return `${p.h12}:${pad(p.m)}${seconds ? ':' + pad(p.s) : ''} ${p.ampm}`;
  }
  function tick(ts) {
    const p = parts(ts);
    return p.m === 0 ? `${p.h12} ${p.ampm}` : `${p.h12}:${pad(p.m)} ${p.ampm}`;
  }
  function dayTime(ts, seconds = true) {
    const p = parts(ts);
    return `${MONTHS[p.d.getMonth()]} ${p.d.getDate()}, ${clock(p.d, seconds)}`;
  }
  const range = (a, b, seconds = true) => `${clock(a, seconds)}–${clock(b, seconds)}`;
  const api = { clock, tick, dayTime, range };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.VLTime = api;
})(typeof window !== 'undefined' ? window : globalThis);
