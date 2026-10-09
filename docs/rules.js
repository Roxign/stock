// Rule lab engine: mirrors stocklab/rules.py (semantics documented there); conditions in rule_catalog.json.
import { twKdj, macd, sma } from "./engine.js?v=6";

let CAT = null;
export async function loadCatalog(url = "rule_catalog.json") {
  if (!CAT) setCatalog(await fetch(url, { cache: "no-cache" }).then((r) => r.json()));
  return CAT;
}
export const catalog = () => CAT;
export function setCatalog(data) {
  CAT = { groups: data.groups, list: data.conditions, byId: Object.fromEntries(data.conditions.map((c) => [c.id, c])) };
  return CAT;
}

const prev = (x, n = 1) => {
  const out = new Float64Array(x.length).fill(NaN);
  for (let i = n; i < x.length; i++) out[i] = x[i - n];
  return out;
};
const map2 = (a, b, f) => { const out = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) out[i] = f(a[i], b[i], i) ? 1 : 0; return out; };
const turn = (x, up) => { const x1 = prev(x), x2 = prev(x, 2); return map2(x, x1, (v, v1, i) => (up ? v > v1 && v1 <= x2[i] : v < v1 && v1 >= x2[i])); };
const cross = (a, b, up) => { const a1 = prev(a), b1 = prev(b); return map2(a, b, (v, w, i) => (up ? v > w && a1[i] <= b1[i] : v < w && a1[i] >= b1[i])); };
const cmp = (x, f) => { const out = new Uint8Array(x.length); for (let i = 0; i < x.length; i++) out[i] = f(x[i], i) ? 1 : 0; return out; };

export function indicators(stock, kdjN = 9, mp = [12, 26, 9]) {
  const kd = twKdj(stock, kdjN), m = macd(Float64Array.from(stock.c), ...mp);
  const close = Float64Array.from(stock.c), ma = new Map();
  return { k: kd.k, d: kd.d, j: kd.j, dif: m.dif, sig: m.macd, osc: m.osc, close, ma: (n) => (ma.has(n) || ma.set(n, sma(close, n)), ma.get(n)) };
}

const param = (c, name) => {
  const spec = CAT.byId[c.id]?.params?.[name];
  return Number(c[name] ?? spec?.default ?? NaN);
};

export function condition(ind, c) {
  const { k, d, j, dif, sig, osc, close } = ind, id = c.id;
  switch (id) {
    case "kd_golden": return cross(k, d, true);
    case "kd_dead": return cross(k, d, false);
    case "k_up": case "k_down": return turn(k, id === "k_up");
    case "j_up": case "j_down": return turn(j, id === "j_up");
    case "kd_bull": return map2(k, d, (a, b) => a > b);
    case "kd_bear": return map2(k, d, (a, b) => a < b);
    case "k_below": case "k_above": case "d_below": case "d_above": case "j_below": case "j_above": {
      const x = { k, d, j }[id[0]], v = param(c, "v");
      return id.endsWith("below") ? cmp(x, (a) => a < v) : cmp(x, (a) => a > v);
    }
    case "macd_golden": return cross(dif, sig, true);
    case "macd_dead": return cross(dif, sig, false);
    case "osc_up": case "osc_down": return turn(osc, id === "osc_up");
    case "osc_rising": { const o1 = prev(osc); return cmp(osc, (a, i) => a > o1[i]); }
    case "osc_falling": { const o1 = prev(osc); return cmp(osc, (a, i) => a < o1[i]); }
    case "dif_up": case "dif_down": return turn(dif, id === "dif_up");
    case "osc_pos": return cmp(osc, (a) => a > 0);
    case "osc_neg": return cmp(osc, (a) => a < 0);
    case "dif_pos": return cmp(dif, (a) => a > 0);
    case "dif_neg": return cmp(dif, (a) => a < 0);
    case "zone_pos": return map2(dif, sig, (a, b) => a > 0 && b > 0);
    case "zone_neg": return map2(dif, sig, (a, b) => a < 0 && b < 0);
    case "close_above_ma": case "close_below_ma": {
      const m = ind.ma(param(c, "n"));
      return id === "close_above_ma" ? map2(close, m, (a, b) => a > b) : map2(close, m, (a, b) => a < b);
    }
    case "ma_rising": case "ma_falling": {
      const m = ind.ma(param(c, "n")), m1 = prev(m);
      return id === "ma_rising" ? map2(m, m1, (a, b) => a > b) : map2(m, m1, (a, b) => a < b);
    }
    default: throw new Error(`unknown condition ${id}`);
  }
}

function recent(a, w) {
  if (w <= 1) return a;
  const out = new Uint8Array(a.length);
  let last = -1;
  for (let i = 0; i < a.length; i++) { if (a[i]) last = i; out[i] = last >= 0 && i - last < w ? 1 : 0; }
  return out;
}

export function sideSignal(ind, side, n) {
  const conds = side?.conds || [];
  const out = new Uint8Array(n);
  if (!conds.length) return out;
  const arrs = conds.map((c) => condition(ind, c));
  if ((side.mode || "all") === "any") {
    for (const a of arrs) for (let i = 0; i < n; i++) out[i] |= a[i];
    return out;
  }
  const w = Math.max(1, Math.floor(side.within || 1));
  const rs = arrs.map((a) => recent(a, w));
  for (let i = 0; i < n; i++) out[i] = rs.every((a) => a[i]) ? 1 : 0;
  return out;
}

// 0/1 target per bar (decided at the close, filled at the next open by engine.simulate).
export function ruleTarget(stock, rule, ind) {
  ind ??= indicators(stock, rule.kdj_n || 9, rule.macd || [12, 26, 9]);
  const n = stock.c.length, o = stock.o, c = ind.close;
  const buy = sideSignal(ind, rule.buy, n), sell = sideSignal(ind, rule.sell, n);
  const stop = (rule.stop || 0) / 100, take = (rule.take || 0) / 100, trail = (rule.trail || 0) / 100, hold = Math.floor(rule.hold || 0);
  const out = new Float64Array(n);
  let on = false, sig = -1, entry = NaN, peak = -Infinity;
  for (let t = 0; t < n; t++) {
    if (!on) {
      if (buy[t]) { on = true; sig = t; entry = NaN; peak = -Infinity; }
    } else {
      if (t === sig + 1) entry = o[t];
      peak = Math.max(peak, c[t]);
      let ex = !!sell[t];
      if (stop && c[t] < entry * (1 - stop)) ex = true;
      if (take && c[t] > entry * (1 + take)) ex = true;
      if (trail && c[t] < peak * (1 - trail)) ex = true;
      if (hold && t - sig >= hold) ex = true;
      if (ex) on = false;
    }
    out[t] = on ? 1 : 0;
  }
  return out;
}

// ---------- text ----------
export function condText(c) {
  const spec = CAT.byId[c.id];
  if (!spec) return c.id;
  return spec.label.replace(/\{(\w+)\}/g, (_, p) => String(c[p] ?? spec.params?.[p]?.default));
}

export function describe(rule) {
  const side = (s, what) => {
    const conds = s?.conds || [];
    if (!conds.length) return what === "賣出" ? "賣出：無（只靠出場保護）" : "買進：無";
    if ((s.mode || "all") === "any") return `${what}：任一符合 — ${conds.map(condText).join("、")}`;
    const w = Math.floor(s.within || 1);
    return `${what}：${w === 1 ? "同一根K棒全部符合" : `最近 ${w} 根K棒內都出現過`} — ${conds.map(condText).join("、")}`;
  };
  const risk = [["stop", "收盤停損"], ["take", "收盤停利"], ["trail", "移動停損"]].filter(([k]) => rule[k]).map(([k, l]) => `${l} ${rule[k]}%`);
  if (rule.hold) risk.push(`最多持有 ${rule.hold} 天`);
  const mp = rule.macd || [12, 26, 9];
  return [side(rule.buy, "買進"), side(rule.sell, "賣出"), "出場保護：" + (risk.join("、") || "無"), `KDJ(${rule.kdj_n || 9},3,3)、MACD(${mp.join(",")})`].join("；") + "。收盤判斷，下一個交易日開盤成交，只做多。";
}

// ---------- share links ----------
export function encodeRule(rule) {
  const bytes = new TextEncoder().encode(JSON.stringify(rule));
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
export function decodeRule(text) {
  try {
    const s = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(s, (ch) => ch.charCodeAt(0))));
  } catch { return null; }
}

// ---------- MultiCharts PowerLanguage ----------
const plCond = (c) => {
  const spec = CAT.byId[c.id];
  return "(" + spec.pl.replace(/\{(\w+)\}/g, (_, p) => String(c[p] ?? spec.params?.[p]?.default)) + ")";
};

export function powerLanguage(rule) {
  const mp = rule.macd || [12, 26, 9];
  const vars = [];
  const sideCode = (s, name, prefix) => {
    const conds = s?.conds || [];
    if (!conds.length) return `${name} = false;`;
    const lines = conds.map((c, i) => { vars.push(`${prefix}${i + 1}(false)`); return `${prefix}${i + 1} = ${plCond(c)};  { ${condText(c)} }`; });
    const w = Math.floor(s.within || 1);
    const names = conds.map((_, i) => `${prefix}${i + 1}`);
    const expr = (s.mode || "all") === "any" ? names.join(" or ") : w === 1 ? names.join(" and ") : names.map((v) => `CountIF(${v}, ${w}) > 0`).join(" and ");
    return lines.join("\n") + `\n${name} = ${expr};`;
  };
  const buyCode = sideCode(rule.buy, "buySig", "b"), sellCode = sideCode(rule.sell, "sellSig", "s");
  return `{ ================================================================================
  規則實驗室匯出：${rule.name || "未命名規則"}
  ${describe(rule)}
  --------------------------------------------------------------------------------
  與網頁及 Python 回測相同：日K，每根K棒收盤後判斷，下一根K棒開盤以市價成交；只做多，部位 0 或 100%。
  進場價 = 買進訊號隔天的開盤價；停損、停利、移動停損都以「收盤價」確認，觸發後下一根開盤賣出
  （請不要改用 SetStopLoss，盤中觸價出場會與回測不同）。出場後要重新出現買進訊號才會再進場。
  KDJ（台灣 ${rule.kdj_n || 9},3,3，1/3 平滑，初值 50；J = 3K - 2D）；MACD：DIF = EMA${mp[0]} - EMA${mp[1]}，MACD 線 = DIF 的 EMA${mp[2]}，柱狀體 OSC = DIF - MACD 線。
  請在 Strategy Properties 設定手續費（0.1425%）、賣出證交稅（股票 0.3%、ETF 0.1%）與滑價；
  Python 回測使用還原權值日線，MultiCharts 若用未還原資料，除權息缺口會讓指標與訊號略有不同。
  回測結果不代表未來績效，也不是投資建議。
  ================================================================================ }

Inputs:
	KLen(${rule.kdj_n || 9}),
	MacdFast(${mp[0]}),
	MacdSlow(${mp[1]}),
	MacdSig(${mp[2]}),
	StopPct(${rule.stop || 0}),
	TakePct(${rule.take || 0}),
	TrailPct(${rule.trail || 0}),
	MaxHold(${rule.hold || 0}),
	Capital(1000000);

Vars:
	rsv(50), kv(50), dv(50), jv(50), hh(0), ll(0),
	dif(0), macdv(0), osc(0),
${vars.length ? "\t" + vars.join(", ") + ",\n" : ""}	buySig(false), sellSig(false),
	inTrade(false), barsIn(0), entryPx(0), peakC(0), exitNow(false), shs(0);

{ ---- KDJ ---- }
hh = Highest(High, KLen);
ll = Lowest(Low, KLen);
if hh - ll <> 0 then
	rsv = (Close - ll) / (hh - ll) * 100
else
	rsv = 50;
kv = kv[1] * 2 / 3 + rsv / 3;
dv = dv[1] * 2 / 3 + kv / 3;
jv = 3 * kv - 2 * dv;

{ ---- MACD ---- }
dif = XAverage(Close, MacdFast) - XAverage(Close, MacdSlow);
macdv = XAverage(dif, MacdSig);
osc = dif - macdv;

{ ---- 買進條件 ---- }
${buyCode}

{ ---- 賣出條件 ---- }
${sellCode}

{ ---- 狀態機：訊號當根收盤決定，下一根開盤成交 ---- }
if inTrade = false then begin
	if buySig then begin
		inTrade = true;
		barsIn = 0;
	end;
end
else begin
	barsIn = barsIn + 1;
	if barsIn = 1 then begin
		entryPx = Open;
		peakC = Close;
	end
	else
		peakC = MaxList(peakC, Close);
	exitNow = sellSig;
	if StopPct > 0 and Close < entryPx * (1 - StopPct / 100) then exitNow = true;
	if TakePct > 0 and Close > entryPx * (1 + TakePct / 100) then exitNow = true;
	if TrailPct > 0 and Close < peakC * (1 - TrailPct / 100) then exitNow = true;
	if MaxHold > 0 and barsIn >= MaxHold then exitNow = true;
	if exitNow then inTrade = false;
end;

{ ---- 下單：0 / 100% 部位 ---- }
if inTrade and MarketPosition = 0 then begin
	shs = IntPortion((Capital + NetProfit) / Close);
	if shs > 0 then
		Buy ("LE") shs shares next bar at market;
end;

if inTrade = false and MarketPosition = 1 then
	Sell ("LX") next bar at market;
`;
}

// ---------- next-session triggers (same method as stocklab/signals.py) ----------
const GRID = Array.from({ length: 41 }, (_, i) => Math.round((-0.1 + i * 0.005) * 1e4) / 1e4);
function tick(px, etf) {
  if (etf) return px < 50 ? 0.01 : 0.05;
  for (const [lim, t] of [[10, 0.01], [50, 0.05], [100, 0.1], [500, 0.5], [1000, 1]]) if (px < lim) return t;
  return 5;
}
const onTick = (px, etf, up) => { const t = tick(px, etf); return Math.round((up ? Math.ceil(px / t - 1e-9) : Math.floor(px / t + 1e-9)) * t * 100) / 100; };

export function scanRule(stock, rule, etf = false, window = 800) {
  const n = stock.c.length, s0 = Math.max(0, n - window);
  const base = { o: stock.o.slice(s0), h: stock.h.slice(s0), l: stock.l.slice(s0), c: stock.c.slice(s0) };
  const cur = ruleTarget(base, rule).at(-1);
  const c0 = base.c.at(-1);
  const decide = (r) => {
    const x = c0 * (1 + r);
    const st = { o: [...base.o, c0], h: [...base.h, Math.max(c0, x)], l: [...base.l, Math.min(c0, x)], c: [...base.c, x] };
    return ruleTarget(st, rule).at(-1);
  };
  const vals = GRID.map(decide), segs = [];
  let lo = null;
  for (let k = 0; k < GRID.length; k++) {
    if (k === GRID.length - 1) segs.push([lo, null, vals[k]]);
    else if (vals[k + 1] !== vals[k]) {
      let a = GRID[k], b = GRID[k + 1];
      for (let it = 0; it < 8; it++) { const m = (a + b) / 2; if (decide(m) === vals[k]) a = m; else b = m; }
      segs.push([lo, a, vals[k]]);
      lo = b;
    }
  }
  const out = [];
  for (const [rlo, rhi, v] of segs) {
    if (v === cur) continue;
    const pl = rlo == null ? null : onTick(c0 * (1 + rlo), etf, true), ph = rhi == null ? null : onTick(c0 * (1 + rhi), etf, false);
    if (pl != null && ph != null && pl > ph) continue;
    const d = ph != null && c0 > ph ? ph / c0 - 1 : pl != null && c0 < pl ? pl / c0 - 1 : 0;
    out.push({ lo: pl, hi: ph, to: v, a: v > cur ? "買進" : "賣出", side: v > cur ? 1 : -1, d: Math.round(d * 1e4) / 1e4 });
  }
  return { cur, trig: out };
}

// ---------- signal quality (same definition as stocklab/signals.py) ----------
export function quality(stock, target, i0, i1, h = 20, step = 0.25) {
  const o = stock.o, n = o.length, buys = [], sells = [];
  let base = 0, cnt = 0;
  for (let i = i0; i <= i1 - h; i++) { base += o[i + h] / o[i] - 1; cnt++; }
  base = cnt ? base / cnt : NaN;
  for (let i = i0 + 1; i <= Math.min(i1 - h, n - 1 - h); i++) {
    const d = target[i - 1] - (i >= 2 ? target[i - 2] : 0);
    const f = o[i + h] / o[i] - 1 - base;
    if (d >= step) buys.push(f); else if (d <= -step) sells.push(f);
  }
  return { buys, sells };
}
