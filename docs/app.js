import { configure, periodBounds, expandTarget, simulate, simulateDca, metrics, xirr, twKdj, macd, sma, ENGINE } from "./engine.js?v=6";
import { loadCatalog, catalog, ruleTarget, describe as ruleDescribe, encodeRule, decodeRule, powerLanguage, scanRule, quality as ruleQuality } from "./rules.js?v=1";

const LWC = window.LightweightCharts;
const $ = (id) => document.getElementById(id);
const FAMILY_SLOT = { "KDJ+MACD 規則": 1, "KDJ+MACD+深度學習": 2, "趨勢追蹤": 3, "均值回歸": 4, "KDJ+MACD 反彈": 5, "深度學習 2.0": 7 };
const LINE_STYLES = ["solid", "dash", "dot", "longdash", "sparsedot"];
const DASH = { solid: "", dash: "4 3", dot: "1.5 2.5", longdash: "9 3", sparsedot: "1.5 5" };
const PERIOD_SHORT = { full: "全期", is: "樣本內", oos: "樣本外", custom: "自訂" };
const PERIOD_TITLE = { full: "2010 至今", is: "2010–2020：策略開發與調參用", oos: "2021 至今：開發時沒看過的資料，用來驗證", custom: "自選起訖日" };

const MAS = [[5, "週線", "--s1"], [10, "雙週線", "--s2"], [20, "月線", "--s3"], [60, "季線", "--s4"], [120, "半年線", "--s5"], [240, "年線", "--s7"]];

// Pages caches files for 10 minutes; revalidate data so a new build shows up right away.
const FRESH = { cache: "no-cache" };

let S;
const stockCache = new Map();
const state = { tab: "signals", period: "oos", universe: "stocks", metric: "cagr_diff", cvMetric: "sharpe", code: "2330", focus: null, from: "", to: "", log: true, sort: { key: "cagr_med", asc: false },
  sigUniverse: "all", sigView: "all", sigNear: 0.03, sigQPeriod: "oos" };
const TABS = ["signals", "overview", "stock", "lab", "strategies"];
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
const strat = (id) => (id === "lab" ? labStrategy() : S.strategies.find((s) => s.id === id) || cvStrat(id));
const baseId = (id) => id.replace(/~cv$/, "");
const isFold = (p) => S.folds.some((f) => f.id === p);
const KIND_UNIVERSE = { etf: "etfs", extra: "extra", stock: "stocks" };
const universeOf = (code) => KIND_UNIVERSE[S.stocks.find((x) => x.code === code)?.kind] || "stocks";
const inUniverse = (s, u) => isBaseline(s) || (u === "extra" ? !!s.extra : (s.universe || "stocks") === u || s.universe === "all");
const appliesTo = (s, st) => isBaseline(s) || s.lab || !!st.pos?.[s.id];

function cvStrat(id) {
  if (!id?.endsWith("~cv")) return undefined;
  const b = S.strategies.find((s) => s.id === baseId(id));
  return b && { ...b, id, label: `${b.label}（交叉驗證）`, cv: true };
}

function foldOptions(el) {
  el.innerHTML = `<option value="">—</option>` + S.folds.map((f, i) => `<option value="${f.id}">第${i + 1}折 ${esc(f.label)}</option>`).join("");
}
const isBaseline = (s) => s.family === "基準";

function assignStyles() {
  const perFamily = {};
  let nextSlot = 5;
  for (const s of S.strategies) {
    if (s.id === "buy_hold") Object.assign(s, { colorVar: "--ink", style: "solid" });
    else if (s.id === "dca") Object.assign(s, { colorVar: "--muted", style: "dash" });
    else if (s.family === "規則實驗室") Object.assign(s, { colorVar: "--lab", style: LINE_STYLES[((perFamily[s.family] = (perFamily[s.family] ?? -1) + 1) + 1) % LINE_STYLES.length] });
    else {
      if (!(s.family in FAMILY_SLOT)) FAMILY_SLOT[s.family] = Math.min(nextSlot++, 5);
      const k = (perFamily[s.family] = (perFamily[s.family] ?? -1) + 1);
      Object.assign(s, { colorVar: `--s${FAMILY_SLOT[s.family]}`, style: LINE_STYLES[k % LINE_STYLES.length] });
    }
  }
}

const swatch = (s) => `<svg class="swatch-svg" viewBox="0 0 20 6" aria-hidden="true" style="color:var(${s.colorVar})"><line x1="1.5" y1="3" x2="18.5" y2="3" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="${DASH[s.style] || ""}"/></svg>`;

// ---------- routing ----------
function readHash() {
  const [path, q = ""] = (location.hash.slice(2) || "signals").split("?");
  const parts = path.split("/");
  const p = new URLSearchParams(q);
  state.tab = TABS.includes(parts[0]) ? parts[0] : "signals";
  if (parts[0] === "stock" && parts[1] && S.stocks.some((x) => x.code === parts[1])) state.code = parts[1];
  if (p.get("p") in PERIOD_SHORT || isFold(p.get("p"))) state.period = p.get("p");
  if (p.get("s") && strat(p.get("s"))) state.focus = p.get("s");
  if (p.get("m")) state.metric = p.get("m");
  if (p.get("from")) state.from = p.get("from");
  if (p.get("to")) state.to = p.get("to");
  if (p.get("r")) {
    const r = decodeRule(p.get("r"));
    if (r?.buy) { labState.rule = r; labTargets.clear(); saveLab(); }
  }
}

function writeHash() {
  const p = new URLSearchParams();
  let path = state.tab;
  if (state.tab === "overview") { p.set("p", ovPeriod()); p.set("m", state.metric); }
  if (state.tab === "lab") p.set("r", encodeRule(labState.rule));
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
  for (const t of TABS) $(`view-${t}`).hidden = t !== state.tab;
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.tab === state.tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (state.tab === "signals") renderSignals();
  else if (state.tab === "lab") renderLab();
  else if (state.tab === "overview") renderOverview();
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
  $("ovFold").value = isFold(p) ? p : "";
  $("ovMetric").value = state.metric;
  seg($("ovUniverse"), Object.entries(S.universes || { stocks: { label: "股票" } }).map(([u, v]) => [u, v.label]), state.universe, (v) => {
    state.universe = v;
    renderOverview();
  });
  renderCv();
  renderPortfolio(p);

  const { key, asc } = state.sort;
  const rows = S.board.filter((r) => r.period === p && (r.universe || "stocks") === state.universe).sort((a, b) => {
    const x = a[key] ?? -Infinity, y = b[key] ?? -Infinity;
    return asc ? x - y : y - x;
  });
  const head = `<thead><tr><th>策略</th>${BOARD_COLS.map(([k, t]) => `<th data-sort="${k}" class="${k === key ? "sorted" + (asc ? " asc" : "") : ""}">${t}</th>`).join("")}</tr></thead>`;
  const body = rows.map((r) => {
    const s = strat(r.id);
    const self = (k) => (r.id === "buy_hold" && k.startsWith("beat_bh")) || (r.id === "dca" && k === "beat_dca") || (r.id === "dca" && ["orders_med", "win_med", "exposure_med"].includes(k));
    return `<tr class="${isBaseline(s) ? "baseline" : ""}"><td><span class="name-cell">${swatch(s)}<a class="link-btn" href="#/strategies?s=${esc(baseId(s.id))}">${esc(s.label)}</a><span class="family">${esc(s.family)}</span></span></td>${BOARD_COLS.map(([k, , f]) => `<td>${self(k) ? "—" : f(r[k])}</td>`).join("")}</tr>`;
  }).join("");
  $("board").innerHTML = head + `<tbody>${body}</tbody>`;
  $("board").querySelectorAll("th[data-sort]").forEach((th) => (th.onclick = () => {
    state.sort = { key: th.dataset.sort, asc: state.sort.key === th.dataset.sort ? !state.sort.asc : false };
    renderOverview();
  }));

  const x = Object.entries(S.metrics.dca[p] || {}).filter(([c]) => universeOf(c) === state.universe).map(([, m]) => m.xirr).filter(isNum).sort((a, b) => a - b);
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
  const cols = S.strategies.filter((s) => s.id !== "buy_hold" && inUniverse(s, state.universe));
  const bhAll = S.metrics.buy_hold[p] || {};
  const head = `<thead><tr><th>${state.universe === "etfs" ? "ETF" : "股票"}</th><th>買進持有<br>年化報酬</th>${cols.map((s) => `<th title="${esc(s.family)}">${swatch(s)}<br>${esc(s.label)}</th>`).join("")}</tr></thead>`;
  const body = S.stocks.filter((st) => universeOf(st.code) === state.universe).map((st) => {
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

const pfShown = new Set(["etf_buy_hold", "etf_dca", "ew_buy_hold"]);
let pfChart = null, pfInit = false;

function pfMeta(id) {
  const lab = S.portfolio?.labels?.[id];
  if (!lab) return strat(id);
  const look = { etf_buy_hold: ["--ink", "solid"], etf_dca: ["--muted", "dash"], ew_buy_hold: ["--ink-2", "dot"] }[id];
  return { id, label: lab, family: "基準", colorVar: look[0], style: look[1], bench: true };
}

function renderPortfolio(p) {
  const P = S.portfolio;
  $("pfSection").hidden = !P || state.universe === "extra";
  if ($("pfSection").hidden) return;
  if (!pfInit) {
    for (const s of S.strategies) if (s.portfolio_weights) pfShown.add(s.id);
    pfInit = true;
  }
  pfChart?.remove();
  pfChart = null;
  const metrics = P.periods[p];
  if (!metrics) {
    $("pfTable").innerHTML = `<tbody><tr><td>投資組合計分板只計算全期、樣本內、樣本外，請切換期間。</td></tr></tbody>`;
    $("lgPf").innerHTML = "";
    return;
  }
  const ids = Object.keys(metrics).filter((id) => P.labels[id] || (strat(id) && !isBaseline(strat(id)) && inUniverse(strat(id), state.universe)));
  ids.sort((a, b) => (metrics[b].sharpe ?? -9) - (metrics[a].sharpe ?? -9));
  const rows = ids.map((id) => {
    const s = pfMeta(id), m = metrics[id];
    return `<tr class="${s.bench ? "baseline" : ""}"><td><span class="name-cell"><input type="checkbox" data-pf="${id}" ${pfShown.has(id) ? "checked" : ""} aria-label="畫出 ${esc(s.label)}">${swatch(s)}${esc(s.label)}${s.portfolio_weights ? ' <span class="family">投資組合策略</span>' : ""}</span></td>
      <td class="${m.cagr >= 0 ? "pos" : "neg"}">${pct(m.cagr)}</td><td>${pct(m.mdd)}</td><td>${num(m.sharpe)}</td><td>${isNum(m.exposure) ? pct(m.exposure, 0) : "—"}</td><td>${isNum(m.turnover) ? pct(m.turnover, 0) : "—"}</td></tr>`;
  }).join("");
  $("pfTable").innerHTML = `<thead><tr><th>投資組合（勾選＝畫線）</th><th>年化報酬</th><th>最大回撤</th><th>Sharpe</th><th>平均持股</th><th>年換手率</th></tr></thead><tbody>${rows}</tbody>`;
  $("pfTable").querySelectorAll("[data-pf]").forEach((cb) => (cb.onchange = () => {
    cb.checked ? pfShown.add(cb.dataset.pf) : pfShown.delete(cb.dataset.pf);
    renderPortfolio(p);
  }));

  pfChart = LWC.createChart($("chPf"), chartOptions(true));
  pfChart.priceScale("right").applyOptions({ mode: LWC.PriceScaleMode.Logarithmic });
  const legend = [];
  for (const id of ids) {
    if (!pfShown.has(id) || !P.equity[p]?.[id]) continue;
    const s = pfMeta(id);
    const ser = pfChart.addLineSeries({ color: cssVar(s.colorVar), lineWidth: 2, lineStyle: LS[s.style], priceLineVisible: false, lastValueVisible: true,
      priceFormat: { type: "custom", formatter: (v) => v.toFixed(2) + "x", minMove: 0.01 } });
    ser.setData(P.equity[p][id].map(([t, v]) => ({ time: t, value: v })));
    legend.push(`<span class="item">${swatch(s)}${esc(s.label)}</span>`);
  }
  pfChart.timeScale().fitContent();
  $("lgPf").innerHTML = `<span><b>資產曲線（週線，期初 = 1.00 倍）</b></span>` + legend.join("");
}

const CV_METRICS = {
  sharpe: { label: "Sharpe", fmt: (x) => num(x), clamp: 0.5 },
  cagr: { label: "年化報酬", fmt: (x) => pct(x), clamp: 0.15 },
  mdd: { label: "最大回撤", fmt: (x) => pct(x), clamp: 0.15 },
};

function renderCv() {
  const T = S.cv_tables?.[state.universe] || S.cv_table;
  $("cvSection").hidden = !T;
  if (!T) return;
  const key = state.cvMetric;
  const M = CV_METRICS[key];
  seg($("cvMetric"), Object.entries(CV_METRICS).map(([k, v]) => [k, v.label]), key, (v) => { state.cvMetric = v; renderCv(); });
  const folds = S.folds;
  const tallyHead = `<th>Sharpe<br>勝折數</th><th>報酬<br>勝折數</th><th>回撤較小<br>折數</th><th>個股×折<br>Sharpe 勝率</th>`;
  const head = `<thead><tr><th>策略</th>${folds.map((f, i) => `<th><button type="button" class="link-btn" data-fold="${f.id}" title="看第${i + 1}折的完整排行">${esc(f.label)}</button></th>`).join("")}${tallyHead}</tr></thead>`;
  const bhRow = `<tr class="baseline"><th>${swatch(strat("buy_hold"))} 買進持有</th>${folds.map((f) => `<td class="ref">${M.fmt(T.bh[f.id][key])}</td>`).join("")}<td class="ref" colspan="4">基準</td></tr>`;
  const rows = T.rows.map((r) => {
    const s = strat(r.id);
    const cells = folds.map((f) => {
      const v = r.folds[f.id]?.[key], b = T.bh[f.id][key];
      if (!isNum(v)) return `<td class="na">—</td>`;
      const d = v - b, w = Math.min(Math.abs(d) / M.clamp, 1) * 85;
      const link = s.cv ? `data-fold="${f.id}"` : `data-fold="${f.id}" data-sid="${s.id}"`;
      return `<td tabindex="0" ${link} title="${esc(s.label)}｜${esc(f.label)}：${M.fmt(v)}（買進持有 ${M.fmt(b)}）" style="background:color-mix(in oklab, var(${d >= 0 ? "--div-pos" : "--div-neg"}) ${w.toFixed(0)}%, var(--div-mid))">${M.fmt(v)}</td>`;
    }).join("");
    const tallies = `<td class="tally">${r.folds_sharpe}/8</td><td class="tally">${r.folds_cagr}/8</td><td class="tally">${r.folds_mdd}/8</td><td class="tally">${pct(r.stock_fold_sharpe, 0)}</td>`;
    return `<tr class="${s.cv ? "cv-row" : ""}"><th>${s.cv ? "" : swatch(s) + " "}${esc(s.cv ? "交叉驗證" + (S.cv[baseId(s.id)]?.mode === "purged" ? "（每折重新訓練）" : "（每折重新選參數）") : s.label)}</th>${cells}${tallies}</tr>`;
  }).join("");
  const t = $("cvTable");
  t.innerHTML = head + `<tbody>${bhRow}${rows}</tbody>`;
  const go = (el) => {
    if (el.dataset.sid) location.hash = `#/stock/${state.code}?s=${el.dataset.sid}&p=${el.dataset.fold}`;
    else { state.period = el.dataset.fold; writeHash(); renderOverview(); $("board").scrollIntoView({ block: "start" }); }
  };
  t.onclick = (e) => { const el = e.target.closest("[data-fold]"); if (el) go(el); };
  t.onkeydown = (e) => { const el = e.target.closest("td[data-fold]"); if (el && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); go(el); } };
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

// ---------- signals ----------
let sigPromise = null, SG = null, fxChart = null;
const loadSignals = () => (sigPromise ??= fetch("data/signals.json", FRESH).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((g) => (SG = g)));
const SIG_VIEWS = [["all", "全部"], ["buy", "買方（買進、回補）"], ["sell", "賣方（賣出、放空）"]];
const NEAR_OPTS = [[0.01, "±1%"], [0.03, "±3%"], [0.05, "±5%"], [0.1, "±10%"]];
const fmtPx = (x) => (isNum(x) ? (x >= 1000 ? x.toLocaleString("en-US", { maximumFractionDigits: 2 }) : String(+x.toFixed(2))) : "—");
const posText = (x) => (Math.abs(x) < 1e-6 ? "空手" : x > 0 ? `多 ${pct(x, 0)}` : `空 ${pct(-x, 0)}`);
const actChip = (a, side) => `<span class="act ${side > 0 ? "buy" : side < 0 ? "sell" : "hold"}">${esc(a)}</span>`;
const sideOk = (side) => state.sigView === "all" || (state.sigView === "buy" ? side > 0 : side < 0);
const trigCond = (t) => (t.lo == null && t.hi == null ? "任何收盤價" : t.lo == null ? `≤ ${fmtPx(t.hi)}` : t.hi == null ? `≥ ${fmtPx(t.lo)}` : `${fmtPx(t.lo)}～${fmtPx(t.hi)}`);
const trigDist = (t) => (t.lo == null && t.hi == null ? "必定觸發" : t.d === 0 ? "平盤即觸發" : spct(t.d));
const trigText = (t) => `收盤 ${trigCond(t)} → ${actChip(t.a, t.side)} <span class="muted">(${trigDist(t)})</span>`;
const sinceText = (x) => (x ? `${x.date} 起 ${posText(x.level)}，進場 ${fmtPx(x.px)}，${x.days} 天，<span class="${x.ret >= 0 ? "pos" : "neg"}">${spct(x.ret)}</span>` : "—");

let follow = null; // Set of followed strategy ids; null = all
try { const f = JSON.parse(localStorage.getItem("follow")); if (Array.isArray(f)) follow = new Set(f); } catch {}
const follows = (sid) => !follow || follow.has(sid);
const saveFollow = () => { try { localStorage.setItem("follow", JSON.stringify(follow ? [...follow] : null)); } catch {} };

function quality(sid, kind, period = "oos") {
  return SG?.strategies[sid]?.quality?.[kind]?.[period];
}

function qualityCell(sid, kind, side) {
  const q = quality(sid, kind);
  if (!q) return "—";
  const v = side > 0 ? q.buy20 : q.sell20;
  const hit = side > 0 ? q.buy_up20 : q.sell_dn20;
  if (!isNum(v)) return "—";
  return `<span class="${v >= 0 ? "pos" : "neg"}">${spct(v)}</span> <span class="muted">${side > 0 ? "漲" : "跌"} ${pct(hit, 0)}</span>`;
}

function renderMacro(m) {
  const box = $("macroCard");
  if (!m) { box.hidden = true; return; }
  box.hidden = false;
  const chg = (x) => `<span class="${x >= 0 ? "pos" : "neg"}">${spct(x)}</span>`;
  const items = [
    ["美元/台幣", m.usdtwd.toFixed(3), `20 日 ${chg(m.usdtwd20)}`],
    ["美元/日圓", m.usdjpy.toFixed(2), `20 日 ${chg(m.usdjpy20)}`],
    ["日圓/台幣", m.jpytwd.toFixed(4), `20 日 ${chg(m.jpytwd20)}`],
    ["聯準會利率", m.fed.toFixed(2) + "%", ""],
    ["日銀利率", m.boj.toFixed(2) + "%", ""],
    ["美日利差", m.diff.toFixed(2), `半年 ${isNum(m.diff120) ? (m.diff120 > 0 ? "+" : "") + m.diff120.toFixed(2) : "—"}`],
  ];
  box.innerHTML = `<div class="macro-items">${items.map(([k, v, sub]) => `<div class="macro-item"><span class="muted">${k}</span><b>${v}</b><span class="muted">${sub}</span></div>`).join("")}
    <div class="macro-item stress ${m.stress ? "on" : ""}"><span class="muted">匯率壓力</span><b>${m.stress ? "有" : "無"}</b><span class="muted">${m.date}</span></div></div>
    <p class="note">${m.stress ? "<strong>注意：</strong>近 20 日日圓對美元升值 ≥ 3% 或台幣貶值 ≥ 1.5%。過去出現時，台股之後的跌幅較深，但平均報酬不一定較差；只當風險提示，不是交易訊號。" : "匯率壓力＝近 20 日日圓對美元升值 ≥ 3%（套利交易平倉）或台幣貶值 ≥ 1.5%（外資匯出）。研究顯示匯率與利差對台股沒有穩定的預測力，只作風險提示。"}
    <button type="button" class="link-btn doc-link" data-doc="research/macro_fx_rates.md" data-title="匯率與美日利差研究">研究報告</button></p>
    <div class="pane-legend" id="lgFx"></div><div class="chart chart-fx" id="chFx"></div>`;
  box.querySelector("[data-doc]").onclick = (e) => openDoc(e.target.dataset.title, e.target.dataset.doc);
  fxChart?.remove();
  fxChart = LWC.createChart($("chFx"), { ...chartOptions(true), leftPriceScale: { visible: true, borderColor: cssVar("--axis") } });
  const h = m.hist;
  const a = fxChart.addLineSeries({ color: cssVar("--s1"), lineWidth: 2, priceScaleId: "left", priceLineVisible: false, lastValueVisible: true });
  const b = fxChart.addLineSeries({ color: cssVar("--s2"), lineWidth: 2, priceScaleId: "right", priceLineVisible: false, lastValueVisible: true });
  a.setData(h.d.map((t, k) => (isNum(h.usdtwd[k]) ? { time: t, value: h.usdtwd[k] } : { time: t })));
  b.setData(h.d.map((t, k) => (isNum(h.usdjpy[k]) ? { time: t, value: h.usdjpy[k] } : { time: t })));
  fxChart.timeScale().fitContent();
  $("lgFx").innerHTML = `<span><b>近一年匯率</b></span><span class="item"><span class="swatch" style="color:var(--s1)"></span>美元/台幣（左軸）</span><span class="item"><span class="swatch" style="color:var(--s2)"></span>美元/日圓（右軸）</span>`;
}

function renderFollowPicker(G) {
  const ids = S.strategies.filter((s) => !isBaseline(s) && G.strategies[s.id]);
  $("sigPickCount").textContent = follow ? `${ids.filter((s) => follow.has(s.id)).length}/${ids.length}` : `全部 ${ids.length}`;
  const box = $("sigPickList");
  if (box.dataset.ready) {
    box.querySelectorAll("[data-follow]").forEach((cb) => (cb.checked = follows(cb.dataset.follow)));
    return;
  }
  const groups = {};
  for (const s of ids) (groups[s.family] ??= []).push(s);
  box.innerHTML = `<div class="actions"><button type="button" class="btn" data-all="1">全選</button><button type="button" class="btn" data-all="0">全不選</button><button type="button" class="btn" data-all="robust" title="買點或賣點在 50 檔、代表股的樣本內外都比平常好的策略">只選訊號穩定的</button></div>` +
    Object.entries(groups).map(([f, list]) => `<fieldset><legend>${esc(f)}</legend>${list.map((s) => `<label class="check"><input type="checkbox" data-follow="${s.id}" ${follows(s.id) ? "checked" : ""}>${swatch(s)}${esc(s.label)}${badges(G.strategies[s.id])}${G.strategies[s.id].triggers ? "" : ' <span class="muted">（無觸發價）</span>'}</label>`).join("")}</fieldset>`).join("");
  box.dataset.ready = "1";
  box.onchange = (e) => {
    const cb = e.target.closest("[data-follow]");
    if (!cb) return;
    follow ??= new Set(ids.map((s) => s.id));
    cb.checked ? follow.add(cb.dataset.follow) : follow.delete(cb.dataset.follow);
    saveFollow();
    renderSignals();
  };
  box.querySelectorAll("[data-all]").forEach((b) => (b.onclick = () => {
    const v = b.dataset.all;
    follow = v === "1" ? null : v === "0" ? new Set() : new Set(ids.filter((s) => G.strategies[s.id].robust_buy || G.strategies[s.id].robust_sell).map((s) => s.id));
    saveFollow();
    renderSignals();
  }));
}

async function renderSignals() {
  const raw = await loadSignals();
  if (state.tab !== "signals") return;
  if (!raw) { $("sigAsof").textContent = "尚未產生訊號資料（執行 python daily_update.py）"; return; }
  const G = raw;
  $("sigAsof").textContent = `${G.asof} 收盤後計算，下一個交易日開盤執行・更新於 ${G.generated.replace("T", " ")}`;
  renderMacro(G.macro);
  seg($("sigUniverse"), [["all", "全部"], ...Object.entries(S.universes).map(([u, v]) => [u, v.label.replace(/（.*/, "")])], state.sigUniverse, (v) => { state.sigUniverse = v; renderSignals(); });
  seg($("sigView"), SIG_VIEWS, state.sigView, (v) => { state.sigView = v; renderSignals(); });
  seg($("sigNear"), NEAR_OPTS, state.sigNear, (v) => { state.sigNear = v; renderSignals(); });
  seg($("sigQPeriod"), [["is", "樣本內 2010–2020"], ["oos", "樣本外 2021–今"]], state.sigQPeriod, (v) => { state.sigQPeriod = v; renderSignals(); });
  renderFollowPicker(G);

  const orders = [], near = [];
  for (const [code, st] of Object.entries(G.stocks)) {
    if (state.sigUniverse !== "all" && st.kind !== state.sigUniverse) continue;
    for (const [sid, e] of Object.entries(st.sig)) {
      const s = strat(sid);
      if (!s || !follows(sid)) continue;
      if (e.side && sideOk(e.side)) orders.push({ code, st, s, e });
      for (const t of e.trig || []) if (Math.abs(t.d) <= state.sigNear + 1e-9 && sideOk(t.side)) near.push({ code, st, s, e, t });
    }
  }
  const order = new Map(S.strategies.map((s, i) => [s.id, i]));
  orders.sort((a, b) => b.e.side - a.e.side || order.get(a.s.id) - order.get(b.s.id) || a.code.localeCompare(b.code));
  near.sort((a, b) => Math.abs(a.t.d) - Math.abs(b.t.d) || a.code.localeCompare(b.code));
  const counts = {};
  for (const o of orders) counts[o.e.a] = (counts[o.e.a] || 0) + 1;
  $("sigCounts").innerHTML = orders.length ? Object.entries(counts).map(([a, n]) => `${actChip(a, orders.find((o) => o.e.a === a).e.side)} ${n}`).join("　") : "";

  const stockCell = (code, st) => `<a class="link-btn" href="#/stock/${code}">${code} ${esc(st.name)}</a>${st.kind === "extra" ? ' <span class="family">代表股</span>' : ""}`;
  const stratCell = (s) => `<span class="name-cell">${swatch(s)}<span>${esc(s.label)}</span></span>`;
  const closeCell = (st) => `${fmtPx(st.close)} <span class="${st.chg >= 0 ? "pos" : "neg"}">${spct(st.chg, 2)}</span>`;
  const link = (code, sid) => `#/stock/${code}?s=${sid}&p=oos`;

  $("sigOrders").innerHTML = orders.length ? `<thead><tr><th>股票</th><th>策略</th><th>明日開盤</th><th>收盤（漲跌）</th><th>部位</th><th>目前部位</th><th>之後的觸發價（明日收盤）</th><th title="樣本外：過去同方向訊號後 20 個交易日，相對同期平均的超額漲跌與上漲／下跌比例">訊號品質</th></tr></thead><tbody>${orders.map(({ code, st, s, e }) => `
    <tr data-href="${link(code, s.id)}"><td>${stockCell(code, st)}</td><td>${stratCell(s)}</td><td>${actChip(e.a, e.side)}${e.warn ? `<br><span class="warn">${esc(e.warn)}</span>` : ""}</td><td>${closeCell(st)}</td>
    <td>${posText(e.held)} → <b>${posText(e.to)}</b></td><td class="wrap">${e.side < 0 || e.held ? sinceText(e.since) : "—"}</td>
    <td class="wrap">${(e.trig || []).slice().sort((a, b) => Math.abs(a.d) - Math.abs(b.d)).slice(0, 2).map(trigText).join("<br>") || (G.strategies[s.id]?.triggers ? "±10% 內無" : "—")}</td>
    <td>${qualityCell(s.id, st.kind, e.side)}</td></tr>`).join("")}</tbody>` : `<tbody><tr><td>目前關注的策略在這個範圍沒有委託。</td></tr></tbody>`;

  $("sigNearTable").innerHTML = near.length ? `<thead><tr><th>股票</th><th>策略</th><th>明日開盤後部位</th><th>明日收盤條件 → 訊號</th><th>距離</th><th>收盤</th><th>訊號品質</th></tr></thead><tbody>${near.map(({ code, st, s, e, t }) => `
    <tr data-href="${link(code, s.id)}"><td>${stockCell(code, st)}</td><td>${stratCell(s)}</td><td>${posText(e.to)}</td><td>收盤 ${trigCond(t)} → ${actChip(t.a, t.side)}</td>
    <td class="${t.d > 0 ? "pos" : t.d < 0 ? "neg" : ""}">${trigDist(t)}</td><td>${closeCell(st)}</td><td>${qualityCell(s.id, st.kind, t.side)}</td></tr>`).join("")}</tbody>` : `<tbody><tr><td>沒有在這個範圍內的觸發價。</td></tr></tbody>`;

  for (const id of ["sigOrders", "sigNearTable"]) $(id).onclick = (ev) => {
    if (ev.target.closest("a")) return;
    const tr = ev.target.closest("tr[data-href]");
    if (tr) location.hash = tr.dataset.href;
  };
  renderQuality(G);
  $("sigDigest").value = digestText(G, orders, near);
}

function renderQuality(G) {
  const kind = state.sigUniverse === "all" ? "stocks" : state.sigUniverse;
  const per = state.sigQPeriod;
  const H = G.horizons || [5, 20];
  const rows = S.strategies.filter((s) => !isBaseline(s) && G.strategies[s.id]?.quality?.[kind]?.[per]).map((s) => {
    const q = G.strategies[s.id].quality[kind][per];
    const ex = (x) => `<td class="${isNum(x) ? (x >= 0 ? "pos" : "neg") : ""}">${spct(x, 2)}</td>`;
    return `<tr class="${follows(s.id) ? "" : "dim"}"><td>${stratCell2(s)}</td><td>${q.codes}</td><td>${num(q.buy_rate, 1)}</td>${H.map((h) => ex(q[`buy${h}`])).join("")}<td>${pct(q.buy_up20, 0)} <span class="muted">/ ${pct(q.base_up20, 0)}</span></td>
      ${H.map((h) => ex(q[`sell${h}`])).join("")}<td>${pct(q.sell_dn20, 0)} <span class="muted">/ ${pct(1 - q.base_up20, 0)}</span></td><td>${q.trades}</td><td>${pct(q.win, 0)}</td><td>${spct(q.avg)}</td><td>${isNum(q.hold) ? Math.round(q.hold) + " 天" : "—"}</td></tr>`;
  }).join("");
  $("sigQuality").innerHTML = `<thead><tr><th>策略</th><th>標的數</th><th>買訊<br>每檔每年</th>${H.map((h) => `<th>買後 ${h} 日<br>超額漲跌</th>`).join("")}<th>買後 20 日上漲<br>比例 / 平常</th>${H.map((h) => `<th>賣後 ${h} 日<br>超額漲跌</th>`).join("")}<th>賣後 20 日下跌<br>比例 / 平常</th><th>完整<br>交易數</th><th>勝率</th><th>平均每筆<br>（含成本）</th><th>平均<br>持有</th></tr></thead><tbody>${rows}</tbody>`;
}
const badges = (m) => (m?.robust_buy ? '<span class="badge buy" title="50 檔、代表股（或 ETF）的樣本內與樣本外，買點後 20 日的超額漲跌都大於 0">買點穩定</span>' : "") +
  (m?.robust_sell ? '<span class="badge sell" title="50 檔、代表股（或 ETF）的樣本內與樣本外，賣點後 20 日的超額漲跌都小於 0">賣點穩定</span>' : "");
const stratCell2 = (s) => `<span class="name-cell">${swatch(s)}<a class="link-btn" href="#/strategies?s=${esc(s.id)}">${esc(s.label)}</a>${badges(SG?.strategies[s.id])}</span>`;

function digestText(G, orders, near) {
  const lines = [`${G.asof} 收盤後訊號（下一個交易日開盤執行）`];
  for (const { code, st, s, e } of orders) lines.push(`【${e.a}】${code} ${st.name}｜${s.label}｜收盤 ${fmtPx(st.close)}${e.warn ? "｜" + e.warn : ""}`);
  if (!orders.length) lines.push("（沒有委託）");
  if (near.length) {
    lines.push("", `接近觸發（明日收盤變動 ${pct(state.sigNear, 0)} 以內）`);
    for (const { code, st, s, t } of near) lines.push(`${code} ${st.name}｜${s.label}｜明日收盤 ${trigCond(t)} → ${t.a}（${trigDist(t)}）`);
  }
  if (G.macro?.stress) lines.push("", "⚠ 匯率壓力：近 20 日日圓急升或台幣急貶");
  return lines.join("\n");
}

async function renderStockSignals(st) {
  const G = await loadSignals();
  const box = $("stSig"), ss = G?.stocks?.[st.code];
  if (!ss || state.tab !== "stock" || st.code !== state.code) { box.hidden = !ss; return; }
  box.hidden = false;
  const sigs = Object.values(ss.sig), nLong = sigs.filter((e) => e.to > 1e-6).length, nShort = sigs.filter((e) => e.to < -1e-6).length;
  $("stSigAsof").innerHTML = `${G.asof} 收盤 ${fmtPx(ss.close)} <span class="${ss.chg >= 0 ? "pos" : "neg"}">${spct(ss.chg, 2)}</span>・明日開盤執行${ss.note_text ? `・下一交易日${esc(ss.note_text)}` : ""}
    ・策略共識（明日開盤後）：<span class="pos">多 ${nLong}</span>・空手 ${sigs.length - nLong - nShort}${nShort ? `・<span class="neg">空 ${nShort}</span>` : ""}`;
  const list = S.strategies.filter((s) => ss.sig[s.id]).map((s) => [s, ss.sig[s.id]]);
  if (labOn()) list.unshift([labStrategy(), labSignal(st)]);
  const rows = list.map(([s, e]) => {
    const trig = (e.trig || []).map(trigText).join("<br>") || (s.lab || G.strategies[s.id]?.triggers ? `<span class="muted">±10% 內無</span>` : `<span class="muted">模型／跨股票決定，無法事先算出</span>`);
    return `<tr class="${s.id === state.focus ? "focus" : ""}"><td><span class="name-cell">${swatch(s)}<button type="button" class="link-btn" data-focus="${s.id}">${esc(s.label)}</button></span></td>
      <td>${actChip(e.a, e.side)}${e.stale ? `<br><span class="warn">部位停在 ${e.stale}</span>` : ""}${e.warn ? `<br><span class="warn">${esc(e.warn)}</span>` : ""}</td><td>${posText(e.held)} → <b>${posText(e.to)}</b></td>
      <td class="wrap">${sinceText(e.since)}</td><td class="wrap">${trig}</td><td>${s.lab ? "—" : qualityCell(s.id, ss.kind, 1)}</td><td>${s.lab ? "—" : qualityCell(s.id, ss.kind, -1)}</td></tr>`;
  }).join("");
  $("stSigTable").innerHTML = `<thead><tr><th>策略</th><th>明日開盤</th><th>部位</th><th>目前部位</th><th>明日收盤觸發價（後天開盤執行）</th><th>買訊品質</th><th>賣訊品質</th></tr></thead><tbody>${rows}</tbody>`;
  $("stSigTable").querySelectorAll("[data-focus]").forEach((b) => (b.onclick = () => setFocus(b.dataset.focus)));
}

// ---------- rule lab ----------
const LAB_ID = "lab";
const LAB_PRESETS = [
  { name: "你的規則：KDJ 與 MACD 零軸下同步翻揚（10% 停損）", buy: { mode: "all", within: 1, conds: [{ id: "k_up" }, { id: "osc_up" }, { id: "zone_neg" }] },
    sell: { mode: "all", within: 1, conds: [{ id: "k_down" }, { id: "osc_down" }, { id: "zone_pos" }] }, stop: 10 },
  { name: "同步翻揚買、MACD 死叉才賣（10% 停損）", buy: { mode: "all", within: 1, conds: [{ id: "k_up" }, { id: "osc_up" }, { id: "zone_neg" }] },
    sell: { mode: "all", within: 1, conds: [{ id: "k_down" }, { id: "macd_dead" }, { id: "zone_pos" }] }, stop: 10 },
  { name: "KDJ 口訣＋MACD 柱確認", buy: { mode: "all", within: 1, conds: [{ id: "j_below", v: 0 }, { id: "osc_rising" }] },
    sell: { mode: "all", within: 1, conds: [{ id: "j_above", v: 100 }, { id: "osc_falling" }] } },
  { name: "KD 黃金交叉買、死亡交叉賣", buy: { mode: "all", within: 1, conds: [{ id: "kd_golden" }] }, sell: { mode: "all", within: 1, conds: [{ id: "kd_dead" }] } },
  { name: "MACD 黃金交叉買、死亡交叉賣", buy: { mode: "all", within: 1, conds: [{ id: "macd_golden" }] }, sell: { mode: "all", within: 1, conds: [{ id: "macd_dead" }] } },
  { name: "規則搜尋第一名（6,930 條中樣本內最佳）", buy: { mode: "all", within: 1, conds: [{ id: "k_up" }, { id: "zone_neg" }] },
    sell: { mode: "all", within: 1, conds: [{ id: "j_above", v: 100 }, { id: "osc_down" }] } },
  { name: "KD 低檔黃金交叉＋MACD 翻揚＋季線上揚（12% 移動停損）", buy: { mode: "all", within: 3, conds: [{ id: "kd_golden" }, { id: "k_below", v: 30 }, { id: "osc_up" }, { id: "ma_rising", n: 60 }] },
    sell: { mode: "all", within: 1, conds: [{ id: "kd_dead" }, { id: "k_above", v: 70 }] }, trail: 12 },
];
const clone = (x) => JSON.parse(JSON.stringify(x));
const ruleKey = (r) => JSON.stringify({ ...r, name: undefined });
const labState = { rule: null, universe: "stocks", period: "oos", sort: { key: "cagr", asc: false }, result: null, used: false, timer: null, running: false };
try {
  const saved = JSON.parse(localStorage.getItem("labRule"));
  if (saved?.buy) labState.rule = saved;
  labState.used = localStorage.getItem("labUsed") === "1";
  labState.chosen = localStorage.getItem("labUsed") !== null;  // the viewer has set the checkbox (or run the lab) before
} catch {}
labState.rule ??= clone(LAB_PRESETS[0]);
const labTargets = new Map();
const labStrategy = () => ({ id: LAB_ID, label: `實驗：${labState.rule.name || "自訂規則"}`, family: "規則實驗室", universe: "all", colorVar: "--lab", style: "solid", lab: true });
const labOn = () => labState.used || state.focus === LAB_ID;
const viewStrategies = () => (labOn() ? [...S.strategies.filter(isBaseline), labStrategy(), ...S.strategies.filter((s) => !isBaseline(s))] : S.strategies);

function labTarget(st) {
  const key = `${st.code}|${ruleKey(labState.rule)}`;
  if (!labTargets.has(key)) {
    if (labTargets.size > 400) labTargets.clear();
    labTargets.set(key, ruleTarget(st, labState.rule));
  }
  return labTargets.get(key);
}

function saveLab() {
  try { localStorage.setItem("labRule", JSON.stringify(labState.rule)); localStorage.setItem("labUsed", labState.used ? "1" : "0"); } catch {}
}

function condOptions(selected) {
  const cat = catalog();
  return cat.groups.map((g) => `<optgroup label="${esc(g)}">${cat.list.filter((c) => c.group === g).map((c) =>
    `<option value="${c.id}" ${c.id === selected ? "selected" : ""}>${esc(c.label.replace(/\{v\}/g, "數值").replace(/\{n\}/g, "N"))}</option>`).join("")}</optgroup>`).join("");
}

function renderLabSide(which) {
  const box = $(which === "buy" ? "labBuy" : "labSell"), side = (labState.rule[which] ??= { mode: "all", within: 1, conds: [] });
  side.conds ??= [];
  const cat = catalog();
  const rows = side.conds.map((c, i) => {
    const spec = cat.byId[c.id], params = Object.entries(spec?.params || {});
    return `<li><select data-i="${i}" data-f="id" aria-label="條件">${condOptions(c.id)}</select>${params.map(([p, ps]) =>
      `<input type="number" data-i="${i}" data-f="${p}" value="${c[p] ?? ps.default}" min="${ps.min}" max="${ps.max}" step="1" aria-label="${p === "n" ? "天數" : "數值"}" title="${p === "n" ? "均線天數" : "門檻值"}">`).join("")}<button type="button" class="icon-btn" data-del="${i}" aria-label="刪除條件">✕</button></li>`;
  }).join("");
  box.innerHTML = `<h3>${which === "buy" ? "買進條件" : "賣出條件"}</h3>
    <div class="lab-row"><div class="seg" data-mode role="group" aria-label="條件組合"></div>
    ${(side.mode || "all") === "all" ? `<label class="field">時間窗<select data-within>${[1, 2, 3, 5, 10].map((w) => `<option value="${w}" ${w === (side.within || 1) ? "selected" : ""}>${w === 1 ? "同一根K棒" : `${w} 根K棒內`}</option>`).join("")}</select></label>` : ""}</div>
    <ol class="cond-list">${rows || `<li class="muted">${which === "sell" ? "沒有賣出條件：只靠右邊的停損、停利出場" : "沒有買進條件：不會進場"}</li>`}</ol>
    <button type="button" class="btn" data-add>＋ 新增條件</button>`;
  seg(box.querySelector("[data-mode]"), [["all", "全部符合"], ["any", "任一符合"]], side.mode || "all", (v) => { side.mode = v; renderLabSide(which); labChanged(); });
  box.querySelector("[data-within]")?.addEventListener("change", (e) => { side.within = +e.target.value; labChanged(); });
  box.querySelector("[data-add]").onclick = () => { side.conds.push({ id: which === "buy" ? "k_up" : "k_down" }); renderLabSide(which); labChanged(); };
  box.querySelectorAll("[data-del]").forEach((b) => (b.onclick = () => { side.conds.splice(+b.dataset.del, 1); renderLabSide(which); labChanged(); }));
  box.querySelectorAll("[data-f]").forEach((el) => (el.onchange = () => {
    const c = side.conds[+el.dataset.i], f = el.dataset.f;
    if (f === "id") {
      side.conds[+el.dataset.i] = { id: el.value };
      renderLabSide(which);
    } else {
      const ps = cat.byId[c.id].params[f];
      c[f] = Math.min(ps.max, Math.max(ps.min, Number(el.value) || ps.default));
      el.value = c[f];
    }
    labChanged();
  }));
}

function renderLabRisk() {
  const r = labState.rule, mp = r.macd || [12, 26, 9];
  const num = (key, label, hint, max) => `<label class="field">${label}<input type="number" data-r="${key}" value="${r[key] || 0}" min="0" max="${max}" step="1" title="${hint}"></label>`;
  $("labRisk").innerHTML = `<h3>出場保護與參數</h3>
    <div class="lab-risk-grid">
      ${num("stop", "收盤停損 %", "收盤跌破進場價這個百分比就賣出；0 = 不用", 50)}
      ${num("take", "收盤停利 %", "收盤漲超過進場價這個百分比就賣出；0 = 不用", 500)}
      ${num("trail", "移動停損 %", "收盤從持有期間最高收盤回落這個百分比就賣出；0 = 不用", 50)}
      ${num("hold", "最長持有天數", "持有滿這麼多個交易日就賣出；0 = 不限", 1000)}
      <label class="field">KDJ 天數<input type="number" data-p="kdj_n" value="${r.kdj_n || 9}" min="2" max="60" step="1"></label>
      <label class="field">MACD 快／慢／訊號<span class="field-row"><input type="number" data-m="0" value="${mp[0]}" min="2" max="100"><input type="number" data-m="1" value="${mp[1]}" min="3" max="200"><input type="number" data-m="2" value="${mp[2]}" min="2" max="100"></span></label>
    </div>`;
  $("labRisk").querySelectorAll("[data-r]").forEach((el) => (el.onchange = () => { r[el.dataset.r] = Math.max(0, Number(el.value) || 0); el.value = r[el.dataset.r]; labChanged(); }));
  $("labRisk").querySelector("[data-p]").onchange = (e) => { r.kdj_n = Math.max(2, Math.round(Number(e.target.value) || 9)); e.target.value = r.kdj_n; labChanged(); };
  $("labRisk").querySelectorAll("[data-m]").forEach((el) => (el.onchange = () => {
    const m = (r.macd ||= [12, 26, 9]).slice();
    m[+el.dataset.m] = Math.max(2, Math.round(Number(el.value) || m[+el.dataset.m]));
    r.macd = m;
    labChanged();
  }));
}

function renderLabMeta() {
  const r = labState.rule;
  $("labName").value = r.name || "";
  $("labDesc").textContent = ruleDescribe(r);
  $("labPl").textContent = powerLanguage(r);
  $("labJson").value = JSON.stringify({ id: "my_rule", name: r.name || "自訂規則", rule: { ...r, name: undefined } }, null, 1);
  const link = `${location.origin}${location.pathname}#/lab?r=${encodeRule(r)}`;
  $("labLink").href = link;
  $("labLink").textContent = link.length > 80 ? link.slice(0, 77) + "…" : link;
}

function labChanged() {
  labTargets.clear();
  saveLab();
  renderLabMeta();
  if (state.tab === "lab") history.replaceState(null, "", `#/lab?r=${encodeRule(labState.rule)}`);
  clearTimeout(labState.timer);
  if (labState.result) labState.timer = setTimeout(runLab, 350);
}

let labReady = false;
async function renderLab() {
  await loadCatalog();
  if (state.tab !== "lab") return;
  if (!labReady) {
    $("labPreset").innerHTML = `<option value="">— 選一個範本 —</option>` + LAB_PRESETS.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("");
    $("labPreset").onchange = (e) => {
      if (e.target.value === "") return;
      labState.rule = clone(LAB_PRESETS[+e.target.value]);
      renderLabSide("buy"); renderLabSide("sell"); renderLabRisk();
      labChanged();
      e.target.value = "";
    };
    $("labName").oninput = (e) => { labState.rule.name = e.target.value; saveLab(); renderLabMeta(); };
    $("labRun").onclick = runLab;
    $("labShow").onchange = (e) => { labState.used = e.target.checked; labState.chosen = true; saveLab(); };
    $("labCopyPl").onclick = async (e) => {
      try { await navigator.clipboard.writeText($("labPl").textContent); e.target.textContent = "已複製"; } catch { e.target.textContent = "請手動選取複製"; }
    };
    labReady = true;
  }
  renderLabSide("buy"); renderLabSide("sell"); renderLabRisk(); renderLabMeta();
  $("labShow").checked = labState.used;
  const unis = [...Object.entries(S.universes).map(([u, v]) => [u, v.label.replace(/（.*/, "")]), ["all", "全部"]];
  seg($("labUniverse"), unis, labState.universe, (v) => { labState.universe = v; renderLab(); if (labState.result) runLab(); });
  seg($("labPeriod"), ["full", "is", "oos"].map((k) => [k, PERIOD_SHORT[k]]), labState.period, (v) => { labState.period = v; renderLab(); if (labState.result) runLab(); });
  if (labState.result) renderLabResult();
}

const labAction = (a, b) => (a === b ? (b > 0 ? ["續抱", 0] : ["空手", 0]) : b > a ? ["買進", 1] : ["賣出", -1]);

async function runLab() {
  if (labState.running) { labState.again = true; return; }
  labState.running = true;
  $("labRun").disabled = true;
  try {
    const u = labState.universe, pkey = labState.period, per = S.periods[pkey];
    const codes = S.stocks.filter((s) => u === "all" || universeOf(s.code) === u).map((s) => s.code);
    let done = 0;
    const queue = codes.filter((c) => !stockCache.has(c));
    const total = queue.length;
    await Promise.all(Array.from({ length: 6 }, async () => {
      while (queue.length) {
        const c = queue.shift();
        await loadStock(c);
        $("labStatus").textContent = `載入股價 ${++done}/${total}…`;
      }
    }));
    $("labStatus").textContent = "計算中…";
    await new Promise((r) => setTimeout(r, 0));
    const rows = [], folds = {};
    for (const code of codes) {
      const st = await loadStock(code), tgt = labTarget(st), n = st.d.length;
      const b = periodBounds(st.d, per.start, per.end);
      const a = labAction(tgt[n - 2], tgt[n - 1]);
      if (!b) continue;
      const r = simulate(st, tgt, b[0], b[1]);
      const m = metrics(r.eq, st.d, b[0], b[1], r.invested, r.trades, r.fills.length);
      const q = ruleQuality(st, tgt, b[0], b[1]);
      rows.push({ code, name: st.name, kind: st.kind, m, bh: S.metrics.buy_hold[pkey]?.[code], q, act: a, trades: r.trades });
      for (const f of S.folds) {
        const fp = S.periods[f.id], fb = periodBounds(st.d, fp.start, fp.end), bh = S.metrics.buy_hold[f.id]?.[code];
        if (!fb || !bh) continue;
        const fr = simulate(st, tgt, fb[0], fb[1]);
        (folds[f.id] ??= []).push([metrics(fr.eq, st.d, fb[0], fb[1]).sharpe, bh.sharpe]);
      }
    }
    labState.result = { rows, folds, universe: u, period: pkey };
    if (!labState.chosen) { labState.used = labState.chosen = true; $("labShow").checked = true; }
    saveLab();
    $("labStatus").textContent = `${rows.length} 檔・${PERIOD_TITLE[pkey]}`;
    renderLabResult();
  } finally {
    labState.running = false;
    $("labRun").disabled = false;
    if (labState.again) { labState.again = false; runLab(); }
  }
}

const median = (v) => { const x = v.filter(isNum).sort((a, b) => a - b); return x.length ? (x.length % 2 ? x[(x.length - 1) / 2] : (x[x.length / 2 - 1] + x[x.length / 2]) / 2) : NaN; };
const mean = (v) => (v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN);

function renderLabResult() {
  const R = labState.result;
  $("labResult").hidden = !R;
  if (!R) return;
  const rows = R.rows.filter((r) => r.bh);
  const med = (f) => median(rows.map(f));
  const beat = (f) => mean(rows.map((r) => (f(r) ? 1 : 0)));
  const trades = rows.flatMap((r) => r.trades);
  const buys = rows.flatMap((r) => r.q.buys), sells = rows.flatMap((r) => r.q.sells);
  const acts = { 1: rows.filter((r) => r.act[1] > 0).length, "-1": rows.filter((r) => r.act[1] < 0).length, hold: rows.filter((r) => r.act[0] === "續抱").length };
  const kpi = (label, v, ref, cls = "") => `<div class="kpi"><span class="muted">${label}</span><b class="${cls}">${v}</b>${ref ? `<span class="ref">${ref}</span>` : ""}</div>`;
  const sign = (x) => (isNum(x) ? (x >= 0 ? "pos" : "neg") : "");
  $("labKpis").innerHTML = [
    kpi("年化報酬（中位數）", pct(med((r) => r.m.cagr)), `買進持有 ${pct(med((r) => r.bh.cagr))}`, sign(med((r) => r.m.cagr))),
    kpi("Sharpe（中位數）", num(med((r) => r.m.sharpe)), `買進持有 ${num(med((r) => r.bh.sharpe))}`),
    kpi("最大回撤（中位數）", pct(med((r) => r.m.mdd)), `買進持有 ${pct(med((r) => r.bh.mdd))}`),
    kpi("勝過買進持有", `${pct(beat((r) => r.m.cagr > r.bh.cagr), 0)}`, `Sharpe 勝過 ${pct(beat((r) => r.m.sharpe > r.bh.sharpe), 0)} 的股票`),
    kpi("平均持股比例", pct(med((r) => r.m.exposure), 0), `每檔交易 ${num(med((r) => r.m.trades), 0)} 次（中位數）`),
    kpi("交易勝率", pct(mean(trades.map((t) => (t.ret > 0 ? 1 : 0))), 0), `平均每筆 ${spct(mean(trades.map((t) => t.ret)))}（含成本）`),
    kpi("買點後 20 日超額漲跌", spct(mean(buys), 2), `${buys.length} 個買點；越高越好`, sign(mean(buys))),
    kpi("賣點後 20 日超額漲跌", spct(mean(sells), 2), `${sells.length} 個賣點；越低越好`, sign(mean(sells))),
    kpi("明日開盤", `${actChip("買進", 1)} ${acts[1]}　${actChip("賣出", -1)} ${acts["-1"]}`, `持有中 ${acts.hold} 檔`),
  ].join("");

  const fids = S.folds.map((f) => f.id);
  const fmed = (f, k) => median((R.folds[f] || []).map((x) => x[k]));
  const won = fids.filter((f) => fmed(f, 0) > fmed(f, 1)).length;
  $("labFolds").innerHTML = `<thead><tr><th></th>${S.folds.map((f) => `<th>${esc(f.label)}</th>`).join("")}<th>Sharpe 勝過折數</th></tr></thead><tbody>
    <tr><td>${swatch(labStrategy())} 本規則</td>${fids.map((f) => { const v = fmed(f, 0), b = fmed(f, 1); return `<td class="${isNum(v) && isNum(b) ? (v > b ? "pos" : "neg") : ""}">${num(v)}</td>`; }).join("")}<td><b>${won}/${fids.length}</b></td></tr>
    <tr class="baseline"><td>${swatch(strat("buy_hold"))} 買進持有</td>${fids.map((f) => `<td>${num(fmed(f, 1))}</td>`).join("")}<td>—</td></tr></tbody>`;

  const cols = [["code", "股票"], ["cagr", "年化報酬"], ["bh_cagr", "買進持有"], ["mdd", "最大回撤"], ["sharpe", "Sharpe"], ["bh_sharpe", "買進持有<br>Sharpe"],
    ["trades", "交易數"], ["win", "勝率"], ["exposure", "持股比例"], ["act", "明日開盤"]];
  const val = { code: (r) => r.code, cagr: (r) => r.m.cagr, bh_cagr: (r) => r.bh.cagr, mdd: (r) => r.m.mdd, sharpe: (r) => r.m.sharpe, bh_sharpe: (r) => r.bh.sharpe,
    trades: (r) => r.m.trades, win: (r) => r.m.win, exposure: (r) => r.m.exposure, act: (r) => r.act[1] * 10 + (r.act[0] === "續抱" ? 1 : 0) };
  const { key, asc } = labState.sort;
  const sorted = rows.slice().sort((a, b) => {
    const x = val[key](a), y = val[key](b);
    const c = typeof x === "string" ? x.localeCompare(y) : (isNum(x) ? x : -Infinity) - (isNum(y) ? y : -Infinity);
    return asc ? c : -c;
  });
  $("labTable").innerHTML = `<thead><tr>${cols.map(([k, t]) => `<th data-sort="${k}" class="${k === key ? "sorted" + (asc ? " asc" : "") : ""}">${t}</th>`).join("")}</tr></thead><tbody>${sorted.map((r) => `
    <tr class="clickable" data-code="${r.code}"><td>${r.code} ${esc(r.name)}${r.kind === "extra" ? ' <span class="family">代表股</span>' : ""}</td>
    <td class="${r.m.cagr > r.bh.cagr ? "pos" : "neg"}">${pct(r.m.cagr)}</td><td>${pct(r.bh.cagr)}</td><td>${pct(r.m.mdd)}</td>
    <td class="${r.m.sharpe > r.bh.sharpe ? "pos" : "neg"}">${num(r.m.sharpe)}</td><td>${num(r.bh.sharpe)}</td><td>${r.m.trades}</td><td>${pct(r.m.win, 0)}</td><td>${pct(r.m.exposure, 0)}</td>
    <td>${actChip(r.act[0], r.act[1])}</td></tr>`).join("")}</tbody>`;
  $("labTable").querySelectorAll("th[data-sort]").forEach((th) => (th.onclick = () => {
    labState.sort = { key: th.dataset.sort, asc: labState.sort.key === th.dataset.sort ? !labState.sort.asc : th.dataset.sort === "code" };
    renderLabResult();
  }));
  $("labTable").onclick = (e) => {
    const tr = e.target.closest("tr[data-code]");
    if (tr) location.hash = `#/stock/${tr.dataset.code}?s=${LAB_ID}&p=${R.period}`;
  };
}

// The lab rule on one stock: today's order, the current position's start and next-session triggers.
function labSignal(st) {
  const tgt = labTarget(st), n = st.d.length;
  const [a, side] = labAction(tgt[n - 2], tgt[n - 1]);
  const { trig } = scanRule(st, labState.rule, st.kind === "etf");
  let since = null;
  if (tgt[n - 2] > 0) {
    let i = n - 1;
    while (i > 1 && tgt[i - 2] > 0) i--;
    since = { date: st.d[i], px: st.o[i], days: n - i, ret: st.c[n - 1] / st.o[i] - 1, level: 1 };
  }
  return { a, side, held: tgt[n - 2], to: tgt[n - 1], since, trig };
}

// ---------- stock view ----------
async function loadStock(code) {
  if (!stockCache.has(code)) {
    stockCache.set(code, fetch(`data/stocks/${code}.json`, FRESH).then((r) => r.json()).then((st) => {
      st.kd = twKdj(st);
      st.ma = Object.fromEntries(MAS.map(([n]) => [n, sma(st.c, n)]));
      st.macd = macd(Float64Array.from(st.c));
      return st;
    }));
  }
  return stockCache.get(code);
}

function initStockControls() {
  const kindLabel = (s) => (s.kind === "etf" ? "ETF" : s.kind === "extra" ? `代表股・${s.sector || ""}` : "0050 成分股");
  $("stockList").innerHTML = S.stocks.map((s) => `<option value="${s.code} ${esc(s.name)}">${esc(kindLabel(s))}</option>`).join("");
  const pick = () => {
    const code = $("stockInput").value.trim().split(/\s+/)[0];
    const hit = S.stocks.find((s) => s.code === code) || S.stocks.find((s) => s.name.includes($("stockInput").value.trim()));
    if (hit && hit.code !== state.code) { state.code = hit.code; writeHash(); renderStock(); }
  };
  $("stockInput").addEventListener("change", pick);
  $("stockInput").addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
  $("stockInput").addEventListener("focus", (e) => e.target.select());
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

// 買/加/賣/減 for long changes; 空/加空/補/減空 for 融券 shorts; 賣空 and 補買 when a fill crosses zero.
function fillLabel(from, to) {
  if (from >= 0 && to >= 0) return to > from ? (from === 0 ? "買" : "加") : to === 0 ? "賣" : "減";
  if (from <= 0 && to <= 0) return to < from ? (from === 0 ? "空" : "加空") : to === 0 ? "補" : "減空";
  return to < 0 ? "賣空" : "補買";
}

function focusOptions(st) {
  const groups = {};
  for (const s of viewStrategies()) if (appliesTo(s, st)) (groups[s.family] ??= []).push(s);
  $("focusSel").innerHTML = Object.entries(groups).map(([f, list]) => `<optgroup label="${esc(f)}">${list.map((s) => `<option value="${s.id}">${esc(s.label)}</option>`).join("")}</optgroup>`).join("");
}

function runAll(st, i0, i1) {
  const out = {};
  for (const s of viewStrategies()) {
    if (!appliesTo(s, st)) continue;
    let r;
    if (s.id === "buy_hold") r = simulate(st, new Float64Array(st.d.length).fill(1), i0, i1);
    else if (s.lab) r = simulate(st, labTarget(st), i0, i1);
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
  $("stFold").value = isFold(state.period) ? state.period : "";

  const [st] = await Promise.all([loadStock(state.code), loadSignals(), loadCatalog()]);
  if (st.code !== state.code || state.tab !== "stock") return;
  if (!appliesTo(strat(state.focus) || {}, st)) {
    state.focus = S.strategies.find((s) => !isBaseline(s) && appliesTo(s, st))?.id || "buy_hold";
    shown.add(state.focus);
  }
  focusOptions(st);
  $("focusSel").value = state.focus;
  $("fromDate").min = $("toDate").min = st.d[0];
  $("fromDate").max = $("toDate").max = st.d[st.d.length - 1];
  if (state.period === "custom") { $("fromDate").value = state.from; $("toDate").value = state.to; }

  const per = state.period === "custom" ? { start: state.from || st.d[0], end: state.to || null } : S.periods[state.period];
  const b = periodBounds(st.d, per.start, per.end);
  $("stockTitle").textContent = `${st.code} ${st.name}${st.sector ? `（${st.sector}）` : ""}`;
  renderStockSignals(st);
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
  const ss = SG?.stocks?.[st.code];
  const focusSig = state.focus === LAB_ID ? labSignal(st) : ss && ss.date === st.d[st.d.length - 1] ? ss.sig[state.focus] : null;
  drawCharts(st, i0, i1, runs, focusSig);
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
    timeScale: { borderColor: cssVar("--axis"), visible: showTime, rightOffset: 4 },
    crosshair: { mode: LWC.CrosshairMode.Normal },
    localization: { locale: "zh-TW", dateFormat: "yyyy-MM-dd" },
    handleScale: { axisPressedMouseMove: { time: true, price: false } },
  };
}

const LS = { solid: 0, dot: 1, dash: 2, longdash: 3, sparsedot: 4 };

function drawCharts(st, i0, i1, runs, sig) {
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
  const markers = fills.map((f) => {
    const buy = f.to > f.from;
    const text = fills.length > 80 ? "" : fillLabel(f.from, f.to);
    return { time: st.d[f.i], position: buy ? "belowBar" : "aboveBar", color: fcolor, shape: buy ? "arrowUp" : "arrowDown", text };
  });
  // Today's decision fills at the next open (not in the data yet); its triggers are tomorrow's closing-price alerts.
  if (sig && i1 === st.d.length - 1) {
    if (sig.side) markers.push({ time: st.d[i1], position: sig.side > 0 ? "belowBar" : "aboveBar", color: sig.side > 0 ? up : down, shape: "circle", text: `明日${sig.a}` });
    for (const t of sig.trig || []) {
      for (const [px, op] of [[t.lo, "≥"], [t.hi, "≤"]]) {
        if (px != null) candle.createPriceLine({ price: px, color: t.side > 0 ? up : down, lineWidth: 1, lineStyle: LWC.LineStyle.Dashed, axisLabelVisible: true, title: `${t.a} ${t.lo != null && t.hi != null ? trigCond(t) : op}` });
      }
    }
  }
  candle.setMarkers(markers);

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
  for (const s of viewStrategies()) {
    if (!shown.has(s.id) || !runs[s.id]) continue;
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
  const rows = viewStrategies().filter((s) => runs[s.id]).map((s) => {
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
          ${s.lab_link ? `<a class="btn" href="${s.lab_link}">在規則實驗室開啟</a>` : ""}
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
    const text = await fetch(btn.dataset.code, FRESH).then((r) => r.text());
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
  const md = await fetch(path, FRESH).then((r) => r.text());
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
  S = await fetch("data/summary.json", FRESH).then((r) => r.json());
  S.folds ??= [];
  S.cv ??= {};
  configure(S.engine);
  assignStyles();
  $("dataDate").textContent = `資料至 ${S.data_end || S.stocks[0].end}・成分股 ${S.data_date}`;
  $("footDate").textContent = S.data_date;
  $("ovMetric").onchange = () => { state.metric = $("ovMetric").value; writeHash(); renderOverview(); };
  foldOptions($("ovFold"));
  foldOptions($("stFold"));
  $("ovFold").onchange = () => { state.period = $("ovFold").value || "oos"; writeHash(); renderOverview(); };
  $("stFold").onchange = () => { state.period = $("stFold").value || "oos"; writeHash(); renderStock(); };
  $("docClose").onclick = () => $("docDialog").close();
  $("sigCopy").onclick = async (e) => {
    try { await navigator.clipboard.writeText($("sigDigest").value); e.target.textContent = "已複製"; } catch { $("sigDigest").select(); e.target.textContent = "請手動複製"; }
  };
  document.querySelectorAll(".callout [data-doc], .doc-link[data-doc]").forEach((b) => (b.onclick = () => openDoc(b.dataset.title, b.dataset.doc)));
  initStockControls();
  readHash();
  $("loading").remove();
  render();
  addEventListener("hashchange", () => { readHash(); render(); });
}

main().catch((e) => { $("loading").textContent = "載入失敗：" + e.message; console.error(e); });
