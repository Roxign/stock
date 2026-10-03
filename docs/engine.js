// Mirrors stocklab/backtest.py and stocklab/indicators.py; keep the two in sync.

export const ENGINE = { capital: 1_000_000, buyFee: 0.001425, sellFee: 0.004425, warmup: 60 };

export function configure(e) {
  ENGINE.capital = e.capital;
  ENGINE.buyFee = e.buy_fee;
  ENGINE.sellFee = e.sell_fee;
  ENGINE.warmup = e.warmup;
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

export function periodBounds(dates, start, end) {
  if (dates.length <= ENGINE.warmup) return null;
  const i0 = Math.max(lowerBound(dates, start), ENGINE.warmup);
  const i1 = end ? upperBound(dates, end) - 1 : dates.length - 1;
  return i1 - i0 >= 20 ? [i0, i1] : null;
}

export function expandTarget(changes, n) {
  const t = new Float64Array(n);
  let k = 0, v = 0;
  for (let i = 0; i < n; i++) {
    while (k < changes.length && changes[k][0] === i) v = changes[k++][1];
    t[i] = v;
  }
  return t;
}

export function simulate(stock, target, i0, i1) {
  const { o, c } = stock, { capital, buyFee, sellFee } = ENGINE;
  const n = i1 - i0 + 1, eq = new Float64Array(n), invested = new Float64Array(n);
  const trades = [], fills = [];
  let cash = capital, sh = 0, cur = 0, entry = null;
  for (let k = 0, i = i0; i <= i1; i++, k++) {
    const t = target[i - 1];
    if (Math.abs(t - cur) > 1e-9) {
      const px = o[i];
      let d = (t * (cash + sh * px)) / px - sh;
      if (d > 0) {
        d = Math.min(d, cash / (px * (1 + buyFee)));
        cash -= d * px * (1 + buyFee);
        sh += d;
      } else if (d < 0) {
        if (t === 0) d = -sh;
        cash += -d * px * (1 - sellFee);
        sh += d;
      }
      fills.push({ i, from: cur, to: t, px });
      if (cur === 0 && t > 0) entry = { i, px };
      else if (cur > 0 && t === 0 && entry) {
        trades.push({ entry: entry.i, exit: i, entryPx: entry.px, exitPx: px, ret: (px * (1 - sellFee)) / (entry.px * (1 + buyFee)) - 1 });
        entry = null;
      }
      cur = t;
    }
    eq[k] = cash + sh * c[i];
    invested[k] = (sh * c[i]) / eq[k];
  }
  if (entry) trades.push({ entry: entry.i, exit: null, entryPx: entry.px, exitPx: c[i1], ret: (c[i1] * (1 - sellFee)) / (entry.px * (1 + buyFee)) - 1 });
  return { eq, invested, trades, fills };
}

export function simulateDca(stock, i0, i1) {
  const { d, o, c } = stock, { capital, buyFee } = ENGINE;
  const n = i1 - i0 + 1, first = new Uint8Array(n);
  let months = 0, prev = "";
  for (let k = 0; k < n; k++) {
    const m = d[i0 + k].slice(0, 7);
    if (m !== prev) { first[k] = 1; months++; prev = m; }
  }
  const amt = capital / months, eq = new Float64Array(n), flows = [], fills = [];
  let cash = capital, sh = 0;
  for (let k = 0, i = i0; i <= i1; i++, k++) {
    if (first[k]) {
      sh += amt / (o[i] * (1 + buyFee));
      cash -= amt;
      flows.push([d[i], -amt]);
      fills.push({ i, from: 0, to: 1, px: o[i] });
    }
    eq[k] = cash + sh * c[i];
  }
  flows.push([d[i1], eq[n - 1] - cash]);
  return { eq, invested: null, trades: [], fills, flows };
}

const DAY = 864e5;

export function xirr(flows) {
  const t0 = Date.parse(flows[0][0]);
  const t = flows.map(([d]) => (Date.parse(d) - t0) / DAY / 365.25);
  const f = flows.map(([, v]) => v);
  const npv = (r) => f.reduce((s, v, k) => s + v / Math.pow(1 + r, t[k]), 0);
  let lo = -0.99, hi = 10;
  if (npv(lo) * npv(hi) > 0) return NaN;
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2;
    if (npv(lo) * npv(mid) <= 0) hi = mid; else lo = mid;
  }
  return (lo + hi) / 2;
}

// invested: per-bar fraction of equity in stock; trades: flat-to-flat round trips; orders: position changes.
export function metrics(eq, dates, i0, i1, invested, trades, orders) {
  const cap = ENGINE.capital, n = eq.length;
  const years = Math.max((Date.parse(dates[i1]) - Date.parse(dates[i0])) / DAY / 365.25, 1 / 365.25);
  const growth = eq[n - 1] / cap;
  let sum = 0, peak = cap, mdd = 0;
  const rets = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    rets[k] = k === 0 ? eq[0] / cap - 1 : eq[k] / eq[k - 1] - 1;
    sum += rets[k];
    peak = Math.max(peak, eq[k]);
    mdd = Math.min(mdd, eq[k] / peak - 1);
  }
  const mean = sum / n;
  let ss = 0;
  for (let k = 0; k < n; k++) ss += (rets[k] - mean) ** 2;
  const sd = Math.sqrt(ss / (n - 1));
  const m = {
    total: growth - 1,
    cagr: Math.pow(growth, 1 / years) - 1,
    mdd,
    sharpe: sd > 0 ? (mean / sd) * Math.sqrt(252) : 0,
    vol: sd * Math.sqrt(252),
    years,
  };
  if (invested) m.exposure = invested.reduce((s, v) => s + v, 0) / n;
  if (orders != null) m.orders = orders;
  if (trades) {
    const r = trades.map((t) => t.ret);
    const closed = trades.filter((t) => t.exit !== null);
    m.trades = trades.length;
    m.win = r.length ? r.filter((x) => x > 0).length / r.length : NaN;
    m.avg_trade = r.length ? r.reduce((s, x) => s + x, 0) / r.length : NaN;
    m.avg_hold = closed.length ? closed.reduce((s, t) => s + t.exit - t.entry, 0) / closed.length : NaN;
  }
  return m;
}

export function sma(x, n) {
  const out = new Float64Array(x.length).fill(NaN);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i];
    if (i >= n) s -= x[i - n];
    if (i >= n - 1) out[i] = s / n;
  }
  return out;
}

export function ema(x, span) {
  const a = 2 / (span + 1), out = new Float64Array(x.length);
  out[0] = x[0];
  for (let i = 1; i < x.length; i++) out[i] = a * x[i] + (1 - a) * out[i - 1];
  return out;
}

export function twKdj(stock, n = 9) {
  const { h, l, c } = stock, len = c.length, k = new Float64Array(len), d = new Float64Array(len), j = new Float64Array(len);
  let kp = 50, dp = 50;
  for (let i = 0; i < len; i++) {
    let hh = -Infinity, ll = Infinity;
    for (let j = Math.max(0, i - n + 1); j <= i; j++) { hh = Math.max(hh, h[j]); ll = Math.min(ll, l[j]); }
    const rsv = hh - ll > 0 ? ((c[i] - ll) / (hh - ll)) * 100 : 50;
    kp = (kp * 2) / 3 + rsv / 3;
    dp = (dp * 2) / 3 + kp / 3;
    k[i] = kp;
    d[i] = dp;
    j[i] = 3 * kp - 2 * dp;
  }
  return { k, d, j };
}

export function macd(close, fast = 12, slow = 26, signal = 9) {
  const ef = ema(close, fast), es = ema(close, slow);
  const dif = ef.map((v, i) => v - es[i]);
  const sig = ema(dif, signal);
  return { dif, macd: sig, osc: dif.map((v, i) => v - sig[i]) };
}
