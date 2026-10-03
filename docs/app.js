import { configure, periodBounds, expandTarget, simulate, simulateDca, metrics, xirr, twKdj, macd, sma, ENGINE } from "./engine.js";

const LWC = window.LightweightCharts;
const $ = (id) => document.getElementById(id);
const FAMILY_SLOT = { "KDJ+MACD 規則": 1, "KDJ+MACD+深度學習": 2, "趨勢追蹤": 3, "均值回歸": 4 };
const LINE_STYLES = ["solid", "dash", "dot"];
const PERIOD_SHORT = { full: "全期", is: "樣本內", oos: "樣本外", custom: "自訂" };
const PERIOD_TITLE = { full: "2010 至今", is: "2010–2020：策略開發與調參用", oos: "2021 至今：開發時沒看過的資料，用來驗證", custom: "自選起訖日" };

const MAS = [[5, "週線", "--s1"], [10, "雙週線", "--s2"], [20, "月線", "--s3"], [60, "季線", "--s4"], [120, "半年線", "--s5"], [240, "年線", "--s7"]];

let S;
const stockCache = new Map();
const state = { tab: "overview", period: "oos", metric: "cagr_diff", code: "2330", focus: null, from: "", to: "", log: true, sort: { key: "cagr_med", asc: false } };
const shown = new Set(["buy_hold", "dca"]);
const shownMa = new Set(MAS.map(([n]) => n));
try { const saved = JSON.parse(localStorage.getItem("mas")); if (Array.isArray(saved)) { shownMa.clear(); saved.forEach((n) => shownMa.add(n)); } } catch {}
let charts = [];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isNum = (x) => typeof x === "number" && isFinite(x);
const pct = (x, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + "%" : "—");
const spct = (x, d = 1) => (isNum(x) ? (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%" : "—");
const num = (x, d = 2) => (isNum(x) ? x.toFixed(d) : "—");
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const strat = (id) => S.strategies.find((s) => s.id === id);
const isBaseline = (s) => s.family === "基準";

function assignStyles() {
  const perFamily = {};
  let nextSlot = 5;
  for (const s of S.strategies) {
    if (s.id === "buy_hold") Object.assign(s, { colorVar: "--ink", style: "solid" });
    else if (s.id === "dca") Object.assign(s, { colorVar: "--muted", style: "dash" });
    else {
      if (!(s.family in FAMILY_SLOT)) FAMILY_SLOT[s.family] = Math.min(nextSlot++, 5);
      const k = (perFamily[s.family] = (perFamily[s.family] ?? -1) + 1);
      Object.assign(s, { colorVar: `--s${FAMILY_SLOT[s.family]}`, style: LINE_STYLES[k % 3] });
    }
  }
}

const swatch = (s) => `<span class="swatch ${s.style === "solid" ? "" : s.style}" style="color:var(${s.colorVar})"></span>`;

// ---------- routing ----------
function readHash() {
  const [path, q = ""] = (location.hash.slice(2) || "overview").split("?");
  const parts = path.split("/");
  const p = new URLSearchParams(q);
  state.tab = ["overview", "stock", "strategies"].includes(parts[0]) ? parts[0] : "overview";
  if (parts[0] === "stock" && parts[1] && S.stocks.some((x) => x.code === parts[1])) state.code = parts[1];
  if (p.get("p") in PERIOD_SHORT) state.period = p.get("p");
  if (p.get("s") && strat(p.get("s"))) state.focus = p.get("s");
  if (p.get("m")) state.metric = p.get("m");
  if (p.get("from")) state.from = p.get("from");
  if (p.get("to")) state.to = p.get("to");
}

function writeHash() {
  const p = new URLSearchParams();
  let path = state.tab;
  if (state.tab === "overview") { p.set("p", ovPeriod()); p.set("m", state.metric); }
  if (state.tab === "stock") {
    path += "/" + state.code;
    p.set("p", state.period);
    if (state.focus) p.set("s", state.focus);
    if (state.period === "custom") { p.set("from", state.from); p.set("to", state.to); }
  }
  history.replaceState(null, "", `#/${path}?${p}`);
}

const ovPeriod = () => (state.period === "custom" ? "oos" : state.period);

function render() {
  for (const t of ["overview", "stock", "strategies"]) $(`view-${t}`).hidden = t !== state.tab;
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.tab === state.tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (state.tab === "overview") renderOverview();
  else if (state.tab === "stock") renderStock();
  else renderStrategies();
}

function seg(el, options, value, onChange) {
  el.replaceChildren(...options.map(([v, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    if (PERIOD_TITLE[v]) b.title = PERIOD_TITLE[v];
    b.setAttribute("aria-pressed", String(v === value));
    b.onclick = () => onChange(v);
    return b;
  }));
}

// ---------- overview ----------
const BOARD_COLS = [
  ["cagr_med", "年化報酬<br>中位數", pct],
  ["cagr_mean", "年化報酬<br>平均", pct],
  ["mdd_med", "最大回撤<br>中位數", pct],
  ["sharpe_med", "Sharpe<br>中位數", num],
  ["exposure_med", "平均<br>持股比例", (x) => pct(x, 0)],
  ["orders_med", "部位<br>調整次數", (x) => num(x, 0)],
  ["win_med", "完整交易<br>勝率", (x) => pct(x, 0)],
  ["beat_bh", "報酬勝過<br>買進持有", (x) => pct(x, 0)],
  ["beat_bh_sharpe", "Sharpe 勝過<br>買進持有", (x) => pct(x, 0)],
  ["beat_dca", "報酬勝過<br>定期定額", (x) => pct(x, 0)],
];

function renderOverview() {
  const p = ovPeriod();
  seg($("ovPeriod"), ["full", "is", "oos"].map((k) => [k, PERIOD_SHORT[k]]), p, (v) => { state.period = v; writeHash(); renderOverview(); });
  $("ovMetric").value = state.metric;

  const { key, asc } = state.sort;
  const rows = S.board.filter((r) => r.period === p).sort((a, b) => {
    const x = a[key] ?? -Infinity, y = b[key] ?? -Infinity;
    return asc ? x - y : y - x;
  });
  const head = `<thead><tr><th>策略</th>${BOARD_COLS.map(([k, t]) => `<th data-sort="${k}" class="${k === key ? "sorted" + (asc ? " asc" : "") : ""}">${t}</th>`).join("")}</tr></thead>`;
  const body = rows.map((r) => {
    const s = strat(r.id);
    const self = (k) => (r.id === "buy_hold" && k.startsWith("beat_bh")) || (r.id === "dca" && k === "beat_dca") || (r.id === "dca" && ["orders_med", "win_med", "exposure_med"].includes(k));
    return `<tr class="${isBaseline(s) ? "baseline" : ""}"><td><span class="name-cell">${swatch(s)}<a class="link-btn" href="#/strategies?s=${esc(s.id)}">${esc(s.label)}</a><span class="family">${esc(s.family)}</span></span></td>${BOARD_COLS.map(([k, , f]) => `<td>${self(k) ? "—" : f(r[k])}</td>`).join("")}</tr>`;
  }).join("");
  $("board").innerHTML = head + `<tbody>${body}</tbody>`;
  $("board").querySelectorAll("th[data-sort]").forEach((th) => (th.onclick = () => {
    state.sort = { key: th.dataset.sort, asc: state.sort.key === th.dataset.sort ? !state.sort.asc : false };
    renderOverview();
  }));

  const x = Object.values(S.metrics.dca[p] || {}).map((m) => m.xirr).filter(isNum).sort((a, b) => a - b);
  $("dcaNote").textContent = x.length ? `定期定額的資金加權報酬率（XIRR，考慮每月分批投入的時間）中位數為 ${pct(x[Math.floor(x.length / 2)])}；表中的年化報酬則以期初就準備好的全部本金計算，讓所有策略用同一個基準比較。` : "";
  renderHeat(p);
}

const HEAT = {
  cagr_diff: { get: (m, bh) => m.cagr - bh.cagr, clamp: 0.2, fmt: spct },
  sharpe_diff: { get: (m, bh) => m.sharpe - bh.sharpe, clamp: 0.6, fmt: (x) => (isNum(x) ? (x > 0 ? "+" : "") + x.toFixed(2) : "—") },
  mdd_diff: { get: (m, bh) => m.mdd - bh.mdd, clamp: 0.25, fmt: spct },
  cagr: { get: (m) => m.cagr, clamp: 0.4, fmt: pct },
};

function renderHeat(p) {
  const H = HEAT[state.metric] || HEAT.cagr_diff;
  const cols = S.strategies.filter((s) => s.id !== "buy_hold");
  const bhAll = S.metrics.buy_hold[p] || {};
  const head = `<thead><tr><th>股票</th><th>買進持有<br>年化報酬</th>${cols.map((s) => `<th title="${esc(s.family)}">${swatch(s)}<br>${esc(s.label)}</th>`).join("")}</tr></thead>`;
  const body = S.stocks.map((st) => {
    const bh = bhAll[st.code];
    const cells = cols.map((s) => {
      const m = S.metrics[s.id]?.[p]?.[st.code];
      if (!m || !bh) return `<td class="na">—</td>`;
      const v = H.get(m, bh);
      const w = Math.min(Math.abs(v) / H.clamp, 1) * 85;
      const pole = v >= 0 ? "--div-pos" : "--div-neg";
      return `<td tabindex="0" data-code="${st.code}" data-sid="${s.id}" style="background:color-mix(in oklab, var(${pole}) ${w.toFixed(0)}%, var(--div-mid))">${H.fmt(v)}</td>`;
    }).join("");
    return `<tr><th>${st.code} ${esc(st.name)}</th><td class="ref">${bh ? pct(bh.cagr) : "—"}</td>${cells}</tr>`;
  }).join("");
  const t = $("heat");
  t.innerHTML = head + `<tbody>${body}</tbody>`;
  const go = (td) => { location.hash = `#/stock/${td.dataset.code}?s=${td.dataset.sid}&p=${p}`; };
  t.onclick = (e) => { const td = e.target.closest("td[data-code]"); if (td) go(td); };
  t.onkeydown = (e) => { const td = e.target.closest("td[data-code]"); if (td && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); go(td); } };
  t.onmousemove = (e) => {
    const td = e.target.closest("td[data-code]");
    if (!td) return hideTip();
    const m = S.metrics[td.dataset.sid][p][td.dataset.code], bh = bhAll[td.dataset.code], st = S.stocks.find((x) => x.code === td.dataset.code);
    showTip(e, `<b>${st.code} ${esc(st.name)}</b><br>${esc(strat(td.dataset.sid).label)}<br>年化 ${pct(m.cagr)}（買進持有 ${pct(bh.cagr)}）<br>Sharpe ${num(m.sharpe)}（${num(bh.sharpe)}）<br>最大回撤 ${pct(m.mdd)}（${pct(bh.mdd)}）${isNum(m.trades) ? `<br>交易 ${m.trades} 次，勝率 ${pct(m.win, 0)}` : ""}`);
  };
  t.onmouseleave = hideTip;
}

function showTip(e, html) {
  const tip = $("tip");
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => ($("tip").hidden = true);

// ---------- stock view ----------
async function loadStock(code) {
  if (!stockCache.has(code)) {
    stockCache.set(code, fetch(`data/stocks/${code}.json`).then((r) => r.json()).then((st) => {
      st.kd = twKdj(st);
      st.ma = Object.fromEntries(MAS.map(([n]) => [n, sma(st.c, n)]));
      st.macd = macd(Float64Array.from(st.c));
      return st;
    }));
  }
  return stockCache.get(code);
}

function initStockControls() {
  $("stockList").innerHTML = S.stocks.map((s) => `<option value="${s.code} ${esc(s.name)}"></option>`).join("");
  const pick = () => {
    const code = $("stockInput").value.trim().split(/\s+/)[0];
    const hit = S.stocks.find((s) => s.code === code) || S.stocks.find((s) => s.name.includes($("stockInput").value.trim()));
    if (hit && hit.code !== state.code) { state.code = hit.code; writeHash(); renderStock(); }
  };
  $("stockInput").addEventListener("change", pick);
  $("stockInput").addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
  $("stockInput").addEventListener("focus", (e) => e.target.select());
  const groups = {};
  for (const s of S.strategies) (groups[s.family] ??= []).push(s);
  $("focusSel").innerHTML = Object.entries(groups).map(([f, list]) => `<optgroup label="${esc(f)}">${list.map((s) => `<option value="${s.id}">${esc(s.label)}</option>`).join("")}</optgroup>`).join("");
  $("focusSel").onchange = () => setFocus($("focusSel").value);
  $("logScale").onchange = () => { state.log = $("logScale").checked; renderStock(); };
  const onDate = () => { state.from = $("fromDate").value; state.to = $("toDate").value; writeHash(); renderStock(); };
  $("fromDate").onchange = onDate;
  $("toDate").onchange = onDate;
}

function setFocus(id) {
  state.focus = id;
  shown.add(id);
  writeHash();
  renderStock();
}

function runAll(st, i0, i1) {
  const out = {};
  for (const s of S.strategies) {
    let r;
    if (s.id === "buy_hold") r = simulate(st, new Float64Array(st.d.length).fill(1), i0, i1);
    else if (s.id === "dca") r = simulateDca(st, i0, i1);
    else r = simulate(st, expandTarget(st.pos[s.id] || [], st.d.length), i0, i1);
    r.m = metrics(r.eq, st.d, i0, i1, r.invested, r.invested ? r.trades : null, r.invested ? r.fills.length : null);
    if (r.flows) r.m.xirr = xirrSafe(r.flows);
    out[s.id] = r;
  }
  return out;
}

const xirrSafe = (f) => { try { return xirr(f); } catch { return NaN; } };

async function renderStock() {
  if (!state.focus) state.focus = S.strategies.find((s) => !isBaseline(s))?.id || "buy_hold";
  shown.add(state.focus);
  const info = S.stocks.find((s) => s.code === state.code);
  $("stockInput").value = `${info.code} ${info.name}`;
  $("focusSel").value = state.focus;
  $("logScale").checked = state.log;
  seg($("stPeriod"), Object.entries(PERIOD_SHORT), state.period, (v) => {
    state.period = v;
    if (v === "custom" && !state.from) { state.from = S.periods.oos.start; state.to = info.end; }
    writeHash();
    renderStock();
  });
  $("customRange").hidden = state.period !== "custom";

  const st = await loadStock(state.code);
  if (st.code !== state.code) return;
  $("fromDate").min = $("toDate").min = st.d[0];
  $("fromDate").max = $("toDate").max = st.d[st.d.length - 1];
  if (state.period === "custom") { $("fromDate").value = state.from; $("toDate").value = state.to; }

  const per = state.period === "custom" ? { start: state.from || st.d[0], end: state.to || null } : S.periods[state.period];
  const b = periodBounds(st.d, per.start, per.end);
  $("stockTitle").textContent = `${st.code} ${st.name}`;
  destroyCharts();
  if (!b) {
    $("stockRange").textContent = `此期間資料不足（${st.name} 的資料從 ${st.d[0]} 開始，需先有 ${ENGINE.warmup} 個交易日暖身）`;
    for (const id of ["stockTable", "tradesTable"]) $(id).innerHTML = "";
    for (const id of ["lgPrice", "lgMa", "lgKd", "lgMacd", "lgEq"]) $(id).innerHTML = "";
    return;
  }
  const [i0, i1] = b;
  $("stockRange").textContent = `${st.d[i0]} ～ ${st.d[i1]}，${i1 - i0 + 1} 個交易日`;
  const runs = runAll(st, i0, i1);
  drawCharts(st, i0, i1, runs);
  renderStockTable(runs);
  renderTrades(st, runs[state.focus]);
}

function destroyCharts() {
  charts.forEach((c) => c.chart.remove());
  charts = [];
}

function chartOptions(showTime) {
  return {
    autoSize: true,
    layout: { background: { type: "solid", color: cssVar("--surface") }, textColor: cssVar("--muted"), fontSize: 11, fontFamily: getComputedStyle(document.body).fontFamily },
    grid: { vertLines: { color: cssVar("--grid") }, horzLines: { color: cssVar("--grid") } },
    rightPriceScale: { borderColor: cssVar("--axis"), minimumWidth: 72 },
    timeScale: { borderColor: cssVar("--axis"), visible: showTime },
    crosshair: { mode: LWC.CrosshairMode.Normal },
    localization: { locale: "zh-TW", dateFormat: "yyyy-MM-dd" },
    handleScale: { axisPressedMouseMove: { time: true, price: false } },
  };
}

const LS = { solid: 0, dot: 1, dash: 2 };

function drawCharts(st, i0, i1, runs) {
  const up = cssVar("--up"), down = cssVar("--down");
  const T = st.d.slice(i0, i1 + 1);
  const idx = new Map(T.map((t, k) => [t, k]));

  const price = LWC.createChart($("chPrice"), chartOptions(false));
  const candle = price.addCandlestickSeries({ upColor: up, downColor: down, borderVisible: false, wickUpColor: up, wickDownColor: down, priceLineVisible: false });
  candle.setData(T.map((t, k) => ({ time: t, open: st.o[i0 + k], high: st.h[i0 + k], low: st.l[i0 + k], close: st.c[i0 + k] })));
  candle.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.2 } });
  const vol = price.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, color: cssVar("--vol"), priceLineVisible: false, lastValueVisible: false });
  vol.setData(T.map((t, k) => ({ time: t, value: st.v[i0 + k] })));
  price.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  candle.priceScale().applyOptions({ mode: state.log ? LWC.PriceScaleMode.Logarithmic : LWC.PriceScaleMode.Normal });
  const maSeries = new Map();
  for (const [n, , v] of MAS) {
    const s = price.addLineSeries({ color: cssVar(v), lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, visible: shownMa.has(n) });
    s.setData(T.map((t, k) => { const x = st.ma[n][i0 + k]; return Number.isFinite(x) ? { time: t, value: x } : { time: t }; }));
    maSeries.set(n, s);
  }
  $("lgMa").innerHTML = MAS.map(([n, name, v]) => `<button type="button" class="ma-tog" data-ma="${n}" aria-pressed="${shownMa.has(n)}"><span class="swatch" style="color:var(${v})"></span>${name} ${n}<b></b></button>`).join("");
  $("lgMa").onclick = (e) => {
    const b = e.target.closest("[data-ma]");
    if (!b) return;
    const n = +b.dataset.ma;
    if (shownMa.has(n)) shownMa.delete(n); else shownMa.add(n);
    b.setAttribute("aria-pressed", String(shownMa.has(n)));
    maSeries.get(n).applyOptions({ visible: shownMa.has(n) });
    try { localStorage.setItem("mas", JSON.stringify([...shownMa])); } catch {}
  };
  const fs = strat(state.focus), fcolor = cssVar(fs.colorVar), fills = runs[state.focus].fills;
  candle.setMarkers(fills.map((f) => {
    const buy = f.to > f.from;
    const text = fills.length > 80 ? "" : buy ? (f.from === 0 ? "買" : "加") : f.to === 0 ? "賣" : "減";
    return { time: st.d[f.i], position: buy ? "belowBar" : "aboveBar", color: fcolor, shape: buy ? "arrowUp" : "arrowDown", text };
  }));

  const kdC = LWC.createChart($("chKd"), chartOptions(false));
  const kS = kdC.addLineSeries({ color: cssVar("--s1"), lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  const dS = kdC.addLineSeries({ color: cssVar("--s2"), lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  const jS = kdC.addLineSeries({ color: cssVar("--s5"), lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  kS.setData(T.map((t, k) => ({ time: t, value: st.kd.k[i0 + k] })));
  dS.setData(T.map((t, k) => ({ time: t, value: st.kd.d[i0 + k] })));
  jS.setData(T.map((t, k) => ({ time: t, value: st.kd.j[i0 + k] })));
  for (const lvl of [20, 80]) kS.createPriceLine({ price: lvl, color: cssVar("--axis"), lineWidth: 1, lineStyle: LWC.LineStyle.Dashed, axisLabelVisible: true, title: "" });

  const mC = LWC.createChart($("chMacd"), chartOptions(true));
  const osc = mC.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
  const difS = mC.addLineSeries({ color: cssVar("--s1"), lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  const sigS = mC.addLineSeries({ color: cssVar("--s2"), lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  osc.setData(T.map((t, k) => ({ time: t, value: st.macd.osc[i0 + k], color: st.macd.osc[i0 + k] >= 0 ? up : down })));
  difS.setData(T.map((t, k) => ({ time: t, value: st.macd.dif[i0 + k] })));
  sigS.setData(T.map((t, k) => ({ time: t, value: st.macd.macd[i0 + k] })));

  const eqC = LWC.createChart($("chEq"), chartOptions(true));
  eqC.priceScale("right").applyOptions({ mode: state.log ? LWC.PriceScaleMode.Logarithmic : LWC.PriceScaleMode.Normal });
  const eqSeries = [];
  for (const s of S.strategies) {
    if (!shown.has(s.id)) continue;
    const ser = eqC.addLineSeries({
      color: cssVar(s.colorVar), lineWidth: s.id === state.focus ? 3 : 2, lineStyle: LS[s.style], priceLineVisible: false, lastValueVisible: true,
      priceFormat: { type: "custom", formatter: (v) => v.toFixed(2) + "x", minMove: 0.01 },
    });
    ser.setData(Array.from(runs[s.id].eq, (v, k) => ({ time: T[k], value: v / ENGINE.capital })));
    eqSeries.push({ s, ser, eq: runs[s.id].eq });
  }

  charts = [
    { chart: price, series: candle, value: (k) => st.c[i0 + k] },
    { chart: kdC, series: kS, value: (k) => st.kd.k[i0 + k] },
    { chart: mC, series: difS, value: (k) => st.macd.dif[i0 + k] },
    { chart: eqC, series: eqSeries[0]?.ser, value: (k) => (eqSeries[0] ? eqSeries[0].eq[k] / ENGINE.capital : 0) },
  ];

  const legends = (k) => {
    const i = i0 + k, prev = i > 0 ? st.c[i - 1] : st.c[i], chg = st.c[i] / prev - 1;
    $("lgPrice").innerHTML = `<span><b>${T[k]}</b></span><span>開 ${num(st.o[i])}</span><span>高 ${num(st.h[i])}</span><span>低 ${num(st.l[i])}</span><span>收 <b>${num(st.c[i])}</b> <span class="${chg >= 0 ? "pos" : "neg"}">${spct(chg, 2)}</span></span><span>量 ${st.v[i].toLocaleString()} 張</span><span class="item">${swatch(fs)}標記：${esc(fs.label)}</span>`;
    $("lgMa").querySelectorAll("[data-ma]").forEach((b) => { b.querySelector("b").textContent = num(st.ma[+b.dataset.ma][i], 1); });
    $("lgKd").innerHTML = `<span>KDJ(9,3,3)</span><span class="item"><span class="swatch" style="color:var(--s1)"></span>K <b>${num(st.kd.k[i], 1)}</b></span><span class="item"><span class="swatch" style="color:var(--s2)"></span>D <b>${num(st.kd.d[i], 1)}</b></span><span class="item"><span class="swatch" style="color:var(--s5)"></span>J <b>${num(st.kd.j[i], 1)}</b></span>`;
    $("lgMacd").innerHTML = `<span>MACD(12,26,9)</span><span class="item"><span class="swatch" style="color:var(--s1)"></span>DIF <b>${num(st.macd.dif[i])}</b></span><span class="item"><span class="swatch" style="color:var(--s2)"></span>MACD <b>${num(st.macd.macd[i])}</b></span><span>OSC <b>${num(st.macd.osc[i])}</b></span>`;
    $("lgEq").innerHTML = `<span><b>資產曲線</b> ${T[k]}</span>` + eqSeries.map(({ s, eq }) => `<span class="item">${swatch(s)}${esc(s.label)} <b>${(eq[k] / ENGINE.capital).toFixed(2)}x</b></span>`).join("");
  };
  legends(T.length - 1);

  let syncing = false;
  for (const c of charts) {
    c.chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
      if (syncing || !r) return;
      syncing = true;
      for (const o of charts) if (o !== c) o.chart.timeScale().setVisibleLogicalRange(r);
      syncing = false;
    });
    c.chart.subscribeCrosshairMove((param) => {
      if (syncing) return;
      syncing = true;
      const k = param.time ? idx.get(param.time) : undefined;
      for (const o of charts) {
        if (o === c || !o.series) continue;
        if (k === undefined) o.chart.clearCrosshairPosition();
        else o.chart.setCrosshairPosition(o.value(k), T[k], o.series);
      }
      legends(k ?? T.length - 1);
      syncing = false;
    });
  }
  price.timeScale().fitContent();
}

function renderStockTable(runs) {
  const cols = ["報酬率", "年化報酬", "最大回撤", "Sharpe", "平均持股", "調整次數", "完整交易", "勝率", "平均每筆", "平均持有"];
  const rows = S.strategies.map((s) => {
    const m = runs[s.id].m, base = isBaseline(s);
    const cagr = s.id === "dca" && isNum(m.xirr) ? `${pct(m.cagr)}<br><span class="muted">XIRR ${pct(m.xirr)}</span>` : pct(m.cagr);
    return `<tr class="${base ? "baseline" : ""} ${s.id === state.focus ? "focus" : ""}">
      <td><span class="name-cell"><input type="checkbox" data-show="${s.id}" ${shown.has(s.id) ? "checked" : ""} aria-label="在資產曲線顯示 ${esc(s.label)}">${swatch(s)}<button type="button" class="link-btn" data-focus="${s.id}">${esc(s.label)}</button><span class="family">${esc(s.family)}</span></span></td>
      <td class="${m.total >= 0 ? "pos" : "neg"}">${pct(m.total)}</td><td>${cagr}</td><td>${pct(m.mdd)}</td><td>${num(m.sharpe)}</td>
      <td>${s.id === "dca" ? "—" : pct(m.exposure, 0)}</td><td>${s.id === "dca" ? "—" : m.orders}</td><td>${s.id === "dca" ? "—" : m.trades}</td><td>${s.id === "dca" ? "—" : pct(m.win, 0)}</td>
      <td>${s.id === "dca" ? "—" : spct(m.avg_trade)}</td><td>${isNum(m.avg_hold) ? Math.round(m.avg_hold) + " 天" : "—"}</td></tr>`;
  }).join("");
  const t = $("stockTable");
  t.innerHTML = `<thead><tr><th>策略（勾選＝畫線）</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  t.querySelectorAll("[data-show]").forEach((cb) => (cb.onchange = () => {
    cb.checked ? shown.add(cb.dataset.show) : shown.delete(cb.dataset.show);
    renderStock();
  }));
  t.querySelectorAll("[data-focus]").forEach((b) => (b.onclick = () => setFocus(b.dataset.focus)));
}

function renderTrades(st, run) {
  const s = strat(state.focus);
  const tr = run.trades;
  $("tradesSummary").textContent = `交易明細：${s.label}（${s.id === "dca" ? run.fills.length + " 次扣款" : tr.length + " 筆"}）`;
  if (s.id === "dca") {
    $("tradesTable").innerHTML = `<tbody><tr><td>定期定額每月第一個交易日開盤買進，不賣出。</td></tr></tbody>`;
    return;
  }
  const rows = tr.slice().reverse().map((t, n) => `<tr><td>${tr.length - n}</td><td>${st.d[t.entry]}</td><td>${num(t.entryPx)}</td><td>${t.exit === null ? "持有中" : st.d[t.exit]}</td><td>${num(t.exitPx)}</td>
    <td class="${t.ret >= 0 ? "pos" : "neg"}">${spct(t.ret)}</td><td>${(t.exit ?? st.d.length - 1) - t.entry} 天</td></tr>`).join("");
  $("tradesTable").innerHTML = `<thead><tr><th>#</th><th>進場日</th><th>進場價</th><th>出場日</th><th>出場價</th><th>報酬（含成本）</th><th>持有</th></tr></thead><tbody>${rows}</tbody>`;
}

// ---------- strategies ----------
function renderStrategies() {
  const el = $("stratList");
  if (!el.dataset.ready) {
    const groups = {};
    for (const s of S.strategies) (groups[s.family] ??= []).push(s);
    el.innerHTML = Object.entries(groups).map(([fam, list]) => `<section class="strat-family"><h2>${esc(fam)}</h2>${list.map((s) => `
      <article class="strat-card" id="card-${s.id}">
        <h3>${swatch(s)}${esc(s.label)} <code class="id">${esc(s.id)}</code></h3>
        <div class="md">${marked.parse(s.description || "")}</div>
        <div class="actions">
          <a class="btn" href="#/stock/${state.code}?s=${s.id}&p=oos">在個股回測中查看</a>
          ${s.multicharts ? `<button class="btn" type="button" data-code="${s.multicharts}" data-sid="${s.id}">MultiCharts 程式碼</button>` : ""}
          ${s.research ? `<button class="btn" type="button" data-doc="${s.research}" data-title="${esc(fam)} 研究筆記">研究筆記</button>` : ""}
        </div>
        <div class="code-box" hidden></div>
      </article>`).join("")}</section>`).join("");
    el.dataset.ready = "1";
    el.querySelectorAll("[data-code]").forEach((b) => (b.onclick = () => toggleCode(b)));
    el.querySelectorAll("[data-doc]").forEach((b) => (b.onclick = () => openDoc(b.dataset.title, b.dataset.doc)));
    el.querySelectorAll(".md a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
  }
  if (state.focus) document.getElementById(`card-${state.focus}`)?.scrollIntoView({ block: "start" });
}

async function toggleCode(btn) {
  const box = btn.closest(".strat-card").querySelector(".code-box");
  if (!box.hidden) { box.hidden = true; return; }
  if (!box.dataset.loaded) {
    const text = await fetch(btn.dataset.code).then((r) => r.text());
    box.innerHTML = `<div class="actions"><button class="btn" type="button">複製程式碼</button><a class="btn" href="${btn.dataset.code}" download="${btn.dataset.sid}.txt">下載 .txt</a></div><pre class="code"><code></code></pre>`;
    box.querySelector("code").textContent = text;
    box.querySelector("button").onclick = async (e) => {
      try { await navigator.clipboard.writeText(text); e.target.textContent = "已複製"; } catch { e.target.textContent = "複製失敗，請手動選取"; }
    };
    box.dataset.loaded = "1";
  }
  box.hidden = false;
}

async function openDoc(title, path) {
  $("docTitle").textContent = title;
  $("docBody").innerHTML = "<p class='muted'>載入中…</p>";
  $("docDialog").showModal();
  const md = await fetch(path).then((r) => r.text());
  $("docBody").innerHTML = marked.parse(md);
  $("docBody").querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
}

// ---------- theme ----------
function effectiveDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

function initTheme() {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch {}
  $("themeBtn").onclick = () => {
    const next = effectiveDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
    if (state.tab === "stock") renderStock();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.tab === "stock") renderStock(); });
}

// ---------- boot ----------
async function main() {
  initTheme();
  S = await fetch("data/summary.json").then((r) => r.json());
  configure(S.engine);
  assignStyles();
  $("dataDate").textContent = `資料至 ${S.stocks[0].end}・成分股 ${S.data_date}`;
  $("footDate").textContent = S.data_date;
  $("ovMetric").onchange = () => { state.metric = $("ovMetric").value; writeHash(); renderOverview(); };
  $("docClose").onclick = () => $("docDialog").close();
  initStockControls();
  readHash();
  $("loading").remove();
  render();
  addEventListener("hashchange", () => { readHash(); render(); });
}

main().catch((e) => { $("loading").textContent = "載入失敗：" + e.message; console.error(e); });
