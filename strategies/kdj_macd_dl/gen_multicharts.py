"""Generate MultiCharts PowerLanguage files for the kdj_macd_dl strategies, embedding a frozen walk-forward model.

  python -m strategies.kdj_macd_dl.gen_multicharts            (run from the repo root)
  python -m strategies.kdj_macd_dl.gen_multicharts --year 2026

By default the model of the most recent walk-forward year is embedded (the same model the backtest uses for that
year: trained on every sample whose label ended before Jan 1 of that year). Re-run each January (after updating the
data) to retrain. After writing, the script verifies the port twice:
  1. features: an independent bar-by-bar Python replica of the PowerLanguage indicator / feature code (KDJ recursion,
     XAverage, Wilder ATR, Average, MA alignment, J base rule) is compared with core.feature_matrix / core.base_rule;
  2. model: the weights are parsed back out of the generated text and the PowerLanguage forward pass (same flat-array
     index arithmetic, same 8-decimal constants) is replayed against the Python model.
"""

import argparse
import re
from pathlib import Path

import numpy as np

from stocklab.data import load_all

from . import core

HERE = Path(__file__).resolve().parent
OUT = HERE / "multicharts"
NAMES = core.feature_names()
N_IN, H1, H2 = len(NAMES), core.HIDDEN[0], core.HIDDEN[1]
N_NET = len(core.SEEDS)
WARM_BARS = 300

# index layout hard-coded in the PowerLanguage code below; must equal the Python feature order
EXPECTED = ([f"{f}_{L}" for L in range(5) for f in ("k", "d")] + [f"j_{L}" for L in range(5)]
            + [f"{f}_{L}" for L in range(5) for f in ("dif", "osc")]
            + [f"{f}_{L}" for L in (0, 5) for f in ("kw", "dw")] + ["jw_0", "jw_5"]
            + [f"{f}_{L}" for L in (0, 5) for f in ("difw", "oscw")]
            + [f"ma_datr_{n}" for n in (5, 10, 20, 60, 120, 240)] + [f"ma_satr_{n}" for n in (20, 60, 120, 240)]
            + ["ma_align", "ma_bull_all", "ma_bear_all"])
assert NAMES == EXPECTED, f"feature order changed; update the PowerLanguage template\n{NAMES}"
assert core.SLOW == 5 and core.MA_N == (5, 10, 20, 60, 120, 240) and core.MA_SLOPE_LAG == 5 and core.SMOOTH == 10
assert core.MA_TIE == 1e-6


def fmt(v):
    s = f"{float(v):.8f}"
    return "0" if float(s) == 0 else s


def flatten(model):
    """Flat arrays exactly as indexed in the PowerLanguage code."""
    W1 = np.zeros(N_NET * H1 * N_IN); B1 = np.zeros(N_NET * H1)
    W2 = np.zeros(N_NET * H2 * H1); B2 = np.zeros(N_NET * H2)
    W3 = np.zeros(N_NET * H2); B3 = np.zeros(N_NET)
    for m, layers in enumerate(model["models"]):
        (w1, b1), (w2, b2), (w3, b3) = layers
        for j in range(H1):
            B1[m * H1 + j] = b1[j]
            for i in range(N_IN):
                W1[m * H1 * N_IN + j * N_IN + i] = w1[j, i]
        for j in range(H2):
            B2[m * H2 + j] = b2[j]
            for i in range(H1):
                W2[m * H2 * H1 + j * H1 + i] = w2[j, i]
        for i in range(H2):
            W3[m * H2 + i] = w3[0, i]
        B3[m] = b3[0]
    return {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3, "NMU": model["mu"], "NSD": model["sd"]}


def array_decl(arrs):
    return ",\n    ".join(f"{k}[{len(v) - 1}](0)" for k, v in arrs.items())


def array_init(arrs):
    lines = []
    for k, v in arrs.items():
        lines.append(f"    {{ {k} }}")
        lines += [f"    {k}[{i}] = {fmt(x)};" for i, x in enumerate(v)]
    return "\n".join(lines)


DAILY = """    { ---------- 台灣 KDJ(9,3,3)：與 stocklab.indicators.tw_kdj 相同（J = 3K - 2D） ---------- }
    hh = Highest(High, 9);
    ll = Lowest(Low, 9);
    if hh - ll <> 0 then rsv = (Close - ll) / (hh - ll) * 100 else rsv = 50;
    kv = kv[1] * 2 / 3 + rsv / 3;
    dv = dv[1] * 2 / 3 + kv / 3;
    jv = 3 * kv - 2 * dv;

    { ---------- MACD(12,26,9) ---------- }
    dif = XAverage(Close, 12) - XAverage(Close, 26);
    sigv = XAverage(dif, 9);
    osc = dif - sigv;
"""

SLOW_AND_RULE = """    { ---------- 週線等效 KDJ(45,15,15)：日 K 上把 9,3,3 乘以 5 ---------- }
    hhw = Highest(High, 45);
    llw = Lowest(Low, 45);
    if hhw - llw <> 0 then rsvw = (Close - llw) / (hhw - llw) * 100 else rsvw = 50;
    kw = kw[1] * 14 / 15 + rsvw / 15;
    dw = dw[1] * 14 / 15 + kw / 15;
    jw = 3 * kw - 2 * dw;

    { ---------- 週線等效 MACD(60,130,45) ---------- }
    difw = XAverage(Close, 60) - XAverage(Close, 130);
    sigw = XAverage(difw, 45);
    oscw = difw - sigw;

    { ---------- 基礎規則（週線等效 KDJ + MACD）----------
      進場：週線等效 J 由 0 以下回到 0 以上（離開超賣區）。
      出場：週線等效 J 由 100 以上跌回 100 以下（離開超買區），但若 MACD 柱狀體 OSC > 0 且仍在放大則續抱。 }
    if baseState = 0 and jw[1] < 0 and jw >= 0 then baseState = 1
    else if baseState = 1 and jw[1] > 100 and jw <= 100 and (oscw <= 0 or oscw <= oscw[1]) then baseState = 0;
"""

MODEL = """    { ---------- ATR(14)，Wilder 平滑：與 stocklab.indicators.atr 相同 ---------- }
    if CurrentBar = 1 then atrv = High - Low
    else atrv = atrv[1] + (TrueRange - atrv[1]) / 14;

    { ---------- 台灣常用均線：週 5、雙週 10、月 20、季 60、半年 120、年 240 ---------- }
    ma5 = Average(Close, 5);
    ma10 = Average(Close, 10);
    ma20 = Average(Close, 20);
    ma60 = Average(Close, 60);
    ma120 = Average(Close, 120);
    ma240 = Average(Close, 240);
    { 均線排列：短均線在長均線之上 +1、之下 -1、相差在百萬分之一以內視為相等 0 }
    alignSum = IFF(ma5 > ma10 * 1.000001, 1, IFF(ma5 < ma10 * 0.999999, -1, 0))
             + IFF(ma10 > ma20 * 1.000001, 1, IFF(ma10 < ma20 * 0.999999, -1, 0))
             + IFF(ma20 > ma60 * 1.000001, 1, IFF(ma20 < ma60 * 0.999999, -1, 0))
             + IFF(ma60 > ma120 * 1.000001, 1, IFF(ma60 < ma120 * 0.999999, -1, 0))
             + IFF(ma120 > ma240 * 1.000001, 1, IFF(ma120 < ma240 * 0.999999, -1, 0));

    { ---------- 48 個輸入特徵（順序必須與訓練時相同） ---------- }
    for ii = 0 to 4 begin
        Xr[2 * ii] = kv[ii] / 100 - 0.5;
        Xr[2 * ii + 1] = dv[ii] / 100 - 0.5;
        Xr[10 + ii] = jv[ii] / 100 - 0.5;
        if atrv[ii] > 0 then begin
            Xr[15 + 2 * ii] = dif[ii] / atrv[ii];
            Xr[16 + 2 * ii] = osc[ii] / atrv[ii];
        end else begin
            Xr[15 + 2 * ii] = NMU[15 + 2 * ii];
            Xr[16 + 2 * ii] = NMU[16 + 2 * ii];
        end;
    end;
    for jj = 0 to 1 begin
        lagw = 5 * jj;
        Xr[25 + 2 * jj] = kw[lagw] / 100 - 0.5;
        Xr[26 + 2 * jj] = dw[lagw] / 100 - 0.5;
        Xr[29 + jj] = jw[lagw] / 100 - 0.5;
        if atrv[lagw] > 0 then begin
            Xr[31 + 2 * jj] = difw[lagw] / atrv[lagw];
            Xr[32 + 2 * jj] = oscw[lagw] / atrv[lagw];
        end else begin
            Xr[31 + 2 * jj] = NMU[31 + 2 * jj];
            Xr[32 + 2 * jj] = NMU[32 + 2 * jj];
        end;
    end;
    { 均線特徵以 ATR 為單位：(收盤 - 均線) / ATR、(均線 - 5 根前均線) / ATR }
    if atrv > 0 then begin
        Xr[35] = (Close - ma5) / atrv;
        Xr[36] = (Close - ma10) / atrv;
        Xr[37] = (Close - ma20) / atrv;
        Xr[38] = (Close - ma60) / atrv;
        Xr[39] = (Close - ma120) / atrv;
        Xr[40] = (Close - ma240) / atrv;
        Xr[41] = (ma20 - ma20[5]) / atrv;
        Xr[42] = (ma60 - ma60[5]) / atrv;
        Xr[43] = (ma120 - ma120[5]) / atrv;
        Xr[44] = (ma240 - ma240[5]) / atrv;
    end else begin
        for ii = 35 to 44 begin
            Xr[ii] = NMU[ii];
        end;
    end;
    Xr[45] = alignSum / 5;                 { 均線排列分數 -1 ~ 1 }
    Xr[46] = IFF(alignSum = 5, 1, 0);      { 完全多頭排列 }
    Xr[47] = IFF(alignSum = -5, 1, 0);     { 完全空頭排列 }

    { ---------- 標準化（訓練集平均/標準差），截斷在 ±5 ---------- }
    for ii = 0 to NIN1 begin
        zz = (Xr[ii] - NMU[ii]) / NSD[ii];
        if zz > 5 then zz = 5;
        if zz < -5 then zz = -5;
        Z[ii] = zz;
    end;

    { ---------- 3 個 MLP（NIN-16-8-1, ReLU, sigmoid）取平均 ---------- }
    prob = 0;
    for mm = 0 to 2 begin
        for jj = 0 to 15 begin
            acc = B1[mm * 16 + jj];
            for ii = 0 to NIN1 begin
                acc = acc + W1[mm * W1STRIDE + jj * NIN + ii] * Z[ii];
            end;
            HA[jj] = IFF(acc > 0, acc, 0);
        end;
        for jj = 0 to 7 begin
            acc = B2[mm * 8 + jj];
            for ii = 0 to 15 begin
                acc = acc + W2[mm * 128 + jj * 16 + ii] * HA[ii];
            end;
            HB[jj] = IFF(acc > 0, acc, 0);
        end;
        acc = B3[mm];
        for ii = 0 to 7 begin
            acc = acc + W3[mm * 8 + ii] * HB[ii];
        end;
        prob = prob + 1 / (1 + ExpValue(-acc));
    end;
    prob = prob / 3;
    probS = XAverage(prob, 10);   { 10 日 EMA 平滑 }
""".replace("NIN1", str(N_IN - 1)).replace("W1STRIDE", str(H1 * N_IN)).replace("NIN", str(N_IN))

ORDERS = """    { ---------- 依目標部位下單：次一根開盤市價成交（與 Python 回測引擎一致） ---------- }
    if CurrentBar > WarmBars and tgt <> lastTgt then begin
        eqv = StartCapital + NetProfit + OpenPositionProfit;
        wantSh = IntPortion(tgt * eqv / Close / LotSize) * LotSize;
        diffSh = wantSh - CurrentShares;
        if tgt = 0 then begin
            if MarketPosition = 1 then Sell ("DL_Exit") next bar at market;
        end
        else if diffSh > 0 then
            Buy ("DL_Add") diffSh shares next bar at market
        else if diffSh < 0 then begin
            sellSh = -diffSh;
            Sell ("DL_Trim") sellSh shares total next bar at market;
        end;
        lastTgt = tgt;
    end;
"""

TARGET = {
    "kdj_macd_dl_floor": """    { ---------- 部位：底倉 0.5；KDJ+MACD 規則持有 或 MLP 相對強 → 1.0 ---------- }
    if baseState = 1 or probS >= QHI then tgt = 1 else tgt = 0.5;
""",
    "kdj_macd_dl_gate": """    { ---------- 部位：KDJ+MACD 規則持有 或 MLP 相對強 → 1；規則空手 且 MLP 相對弱 → 0；其他 → 0.5 ---------- }
    if baseState = 1 or probS >= QHI then tgt = 1
    else if probS < QLO then tgt = 0
    else tgt = 0.5;
""",
}

TITLE = {
    "kdj_macd_dl_floor": "KDJ+MACD+MLP 半倉底倉",
    "kdj_macd_dl_gate": "KDJ+MACD+MLP 三段部位 (0 / 0.5 / 1)",
}

PYRAMID_NOTE = {
    True: """  2. 本策略會加碼與部分減碼，請在 Strategy Properties 勾選允許同方向多筆進場
     (Allow up to N entry orders in the same direction，N 設 2 以上)。""",
    False: "  2. 本策略只有全進全出，不需要允許同方向多筆進場。",
}


def common_header(pyramid, warm):
    return f"""  使用前請注意：
  1. 交易成本請在 Strategy Properties 設定（本程式碼本身不扣任何成本）：
     買進手續費 0.1425%、賣出手續費 0.1425% + 證交稅 0.3%（Python 回測使用的數字），
     滑價請自行設定合理值（例如每筆 1~2 個 tick）；Python 回測假設以次日開盤價成交、沒有額外滑價。
""" + PYRAMID_NOTE[pyramid] + f"""
  3. Input StartCapital 請設成與 Strategy Properties 的 Initial Capital 相同；LotSize = 1 表示可下零股，
     改成 1000 則只下整張。下單股數以訊號當根收盤價估算，Python 回測則以次日開盤價計算，會有些微差異。
  4. 請載入至少 3 年的日 K 資料，Max number of bars study will reference 設 {warm} 以上：
     年線 Average(Close, 240) 加 5 根落後、週線等效 MACD(60,130,45) 都需要很長的暖機期，WarmBars 之前不下單。
  5. 訊號在 K 棒收盤時計算，於「下一根開盤」以市價成交，與網站上 Python 回測的規則一致。"""


def header_dl(sid, year, model):
    p = core.n_params(N_IN)
    return f"""{{ ==========================================================================================
  {sid} — {TITLE[sid]}
  家族：KDJ+MACD+深度學習（由 strategies/kdj_macd_dl/gen_multicharts.py 自動產生，請勿手動修改權重）

  【內嵌模型為凍結版本】
  MultiCharts 無法訓練模型。本檔內嵌的是 Python 前推式訓練中 {year} 年使用的模型：
  訓練樣本 = 50 檔成分股中，60 日標籤在 {year}-01-01 之前已結束的 {model['n_train']:,} 筆資料。
  模型不會自己更新；要用最新資料重新訓練，請在 Python 端更新資料後重新執行
      python -m strategies.kdj_macd_dl.gen_multicharts
  再把新產生的程式碼貼回 MultiCharts 重新編譯。建議每年 1 月更新一次（與回測的重訓頻率相同）。
  回測中每年使用「當年」的模型；本檔整段歷史都用同一個凍結模型，所以在 MultiCharts 回測較早年份時，
  等於用了「未來」訓練出的模型，那段歷史績效不能當真，只有 {year} 年以後才和 Python 回測一致。

  模型：{N_NET} 個多層感知器 (MLP) {N_IN}→{H1}→{H2}→1，每個 {p:,} 個參數，合計 {N_NET * p:,} 個參數，輸出取平均。
  輸入（{N_IN} 個）：日 KDJ(9,3,3) 的 K、D、J（最近 5 根）；日 MACD(12,26,9) 的 DIF、OSC 除以 ATR(14)（最近 5 根）；
        週線等效 KDJ(45,15,15) 的 K、D、J 與 MACD(60,130,45)（今天與 5 根前）；
        5/10/20/60/120/240 日均線：(收盤 - 均線)/ATR、20/60/120/240 日均線 5 日變化/ATR、均線排列。
  輸出：未來 60 個交易日「贏過 0050 成分股中位數」的機率；平滑後與訓練集預測值的
        第 20 百分位 (QLO = {fmt(model['q_lo'])}) / 第 40 百分位 (QHI = {fmt(model['q_hi'])}) 比較。

{common_header(True, WARM_BARS)}
  ========================================================================================== }}
"""


def build_dl(sid, year, model):
    arrs = flatten(model)
    return f"""{header_dl(sid, year, model)}
Inputs:
    StartCapital(1000000),
    LotSize(1),
    WarmBars({WARM_BARS});

Variables:
    hh(0), ll(0), rsv(50), kv(50), dv(50), jv(50),
    hhw(0), llw(0), rsvw(50), kw(50), dw(50), jw(50),
    dif(0), sigv(0), osc(0), difw(0), sigw(0), oscw(0),
    atrv(0), baseState(0),
    ma5(0), ma10(0), ma20(0), ma60(0), ma120(0), ma240(0), alignSum(0),
    ii(0), jj(0), mm(0), lagw(0), zz(0), acc(0), prob(0), probS(0),
    QLO({fmt(model['q_lo'])}), QHI({fmt(model['q_hi'])}),
    tgt(0), lastTgt(-1), eqv(0), wantSh(0), diffSh(0), sellSh(0);

Arrays:
    Xr[{N_IN - 1}](0), Z[{N_IN - 1}](0), HA[{H1 - 1}](0), HB[{H2 - 1}](0),
    {array_decl(arrs)};

{{ ---------- 模型權重與標準化常數：只在第一根 K 棒載入一次 ---------- }}
once begin
{array_init(arrs)}
end;

{DAILY}
{SLOW_AND_RULE}
{MODEL}
{TARGET[sid]}
{ORDERS}"""


def build_base():
    return f"""{{ ==========================================================================================
  kdj_macd_dl_base — 週KDJ+MACD 基礎規則（無 DL）
  家族：KDJ+MACD+深度學習（DL 版本所依附的基礎規則，沒有任何模型，用來對照 DL 是否加分）

  規則：日線資料上使用「週線等效」參數 KDJ(45,15,15)、MACD(60,130,45)，J = 3K - 2D。
        進場：J 由 0 以下回到 0 以上（離開超賣區）→ 次日開盤全數買進。
        出場：J 由 100 以上跌回 100 以下（離開超買區）→ 次日開盤全數賣出；
              但若當時 MACD 柱狀體 OSC > 0 且比前一根大（漲勢仍在加速）則不賣，繼續持有。

{common_header(False, 250)}
  ========================================================================================== }}

Inputs:
    StartCapital(1000000),
    LotSize(1),
    WarmBars(250);

Variables:
    hhw(0), llw(0), rsvw(50), kw(50), dw(50), jw(50),
    difw(0), sigw(0), oscw(0),
    baseState(0), tgt(0), lastTgt(-1), eqv(0), wantSh(0), diffSh(0), sellSh(0);

{SLOW_AND_RULE}
    tgt = baseState;

{ORDERS}"""


# ---------------------------------------------------------------- verification of the generated code

def pl_replica(df):
    """Bar-by-bar scalar re-implementation of the PowerLanguage indicator / feature / base-rule code above
    (independent of pandas / core), returning (X [n, N_IN], baseState [n])."""
    H, L, C = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    n = len(C)
    X = np.full((n, N_IN), np.nan)
    state = np.zeros(n)
    S = {k: np.zeros(n) for k in ("kv", "dv", "jv", "dif", "osc", "kw", "dw", "jw", "difw", "oscw", "atrv",
                                  "ma20", "ma60", "ma120", "ma240")}
    e = {}
    kv = dv = kw = dw = 50.0
    base = 0.0

    def xav(key, price, length, t):
        e[key] = price if t == 0 else e[key] + 2 / (length + 1) * (price - e[key])
        return e[key]

    for t in range(n):
        hh, ll = H[max(0, t - 8) : t + 1].max(), L[max(0, t - 8) : t + 1].min()
        rsv = (C[t] - ll) / (hh - ll) * 100 if hh - ll != 0 else 50
        kv = kv * 2 / 3 + rsv / 3
        dv = dv * 2 / 3 + kv / 3
        dif = xav("e12", C[t], 12, t) - xav("e26", C[t], 26, t)
        osc = dif - xav("s9", dif, 9, t)
        hhw, llw = H[max(0, t - 44) : t + 1].max(), L[max(0, t - 44) : t + 1].min()
        rsvw = (C[t] - llw) / (hhw - llw) * 100 if hhw - llw != 0 else 50
        kw = kw * 14 / 15 + rsvw / 15
        dw = dw * 14 / 15 + kw / 15
        difw = xav("e60", C[t], 60, t) - xav("e130", C[t], 130, t)
        oscw = difw - xav("s45", difw, 45, t)
        tr = H[t] - L[t] if t == 0 else max(H[t], C[t - 1]) - min(L[t], C[t - 1])
        atrv = tr if t == 0 else S["atrv"][t - 1] + (tr - S["atrv"][t - 1]) / 14
        ma = {m: (C[t - m + 1 : t + 1].sum() / m if t >= m - 1 else np.nan) for m in (5, 10, 20, 60, 120, 240)}
        for k, v in (("kv", kv), ("dv", dv), ("jv", 3 * kv - 2 * dv), ("dif", dif), ("osc", osc), ("kw", kw),
                     ("dw", dw), ("jw", 3 * kw - 2 * dw), ("difw", difw), ("oscw", oscw), ("atrv", atrv),
                     ("ma20", ma[20]), ("ma60", ma[60]), ("ma120", ma[120]), ("ma240", ma[240])):
            S[k][t] = v
        if t >= 1:
            jw0, jw1, o0, o1 = S["jw"][t], S["jw"][t - 1], S["oscw"][t], S["oscw"][t - 1]
            if base == 0 and jw1 < 0 and jw0 >= 0:
                base = 1.0
            elif base == 1 and jw1 > 100 and jw0 <= 100 and (o0 <= 0 or o0 <= o1):
                base = 0.0
        state[t] = base
        if t < 250:
            continue
        x = np.zeros(N_IN)
        for ii in range(5):
            x[2 * ii] = S["kv"][t - ii] / 100 - 0.5
            x[2 * ii + 1] = S["dv"][t - ii] / 100 - 0.5
            x[10 + ii] = S["jv"][t - ii] / 100 - 0.5
            x[15 + 2 * ii] = S["dif"][t - ii] / S["atrv"][t - ii]
            x[16 + 2 * ii] = S["osc"][t - ii] / S["atrv"][t - ii]
        for jj in range(2):
            lw = 5 * jj
            x[25 + 2 * jj] = S["kw"][t - lw] / 100 - 0.5
            x[26 + 2 * jj] = S["dw"][t - lw] / 100 - 0.5
            x[29 + jj] = S["jw"][t - lw] / 100 - 0.5
            x[31 + 2 * jj] = S["difw"][t - lw] / S["atrv"][t - lw]
            x[32 + 2 * jj] = S["oscw"][t - lw] / S["atrv"][t - lw]
        for i, m in enumerate((5, 10, 20, 60, 120, 240)):
            x[35 + i] = (C[t] - ma[m]) / atrv
        for i, m in enumerate((20, 60, 120, 240)):
            x[41 + i] = (S[f"ma{m}"][t] - S[f"ma{m}"][t - 5]) / atrv
        sgn = lambda a, b: 1 if a > b * 1.000001 else (-1 if a < b * 0.999999 else 0)
        al = sgn(ma[5], ma[10]) + sgn(ma[10], ma[20]) + sgn(ma[20], ma[60]) + sgn(ma[60], ma[120]) + sgn(ma[120], ma[240])
        x[45], x[46], x[47] = al / 5, float(al == 5), float(al == -5)
        X[t] = x
    return X, state


def verify_features(data, codes):
    worst, worst_rel, nbars, base_diff = 0.0, 0.0, 0, 0
    for code in codes:
        df = data[code]
        ind = core.indicator_frame(df)
        Xp, _ = core.feature_matrix(ind)
        Xr, st = pl_replica(df)
        rows = np.flatnonzero(~np.isnan(Xr).any(axis=1) & ~np.isnan(Xp).any(axis=1))
        d = np.abs(Xr[rows] - Xp[rows].astype(np.float64))
        worst = max(worst, float(d.max()))
        worst_rel = max(worst_rel, float((d / np.maximum(np.abs(Xr[rows]), 1.0)).max()))
        nbars += len(rows)
        base_diff += int((st != core.base_rule(ind)).sum())
    return worst, worst_rel, nbars, base_diff


def parse_arrays(text):
    vals = {}
    for k, i, v in re.findall(r"^\s+(W1|B1|W2|B2|W3|B3|NMU|NSD)\[(\d+)\] = (-?[0-9.]+);", text, flags=re.M):
        vals.setdefault(k, {})[int(i)] = float(v)
    return {k: np.array([d[i] for i in range(len(d))]) for k, d in vals.items()}


def pl_forward(A, x):
    """Python replica of the PowerLanguage loops (flat indices, IFF ReLU, ExpValue sigmoid)."""
    Z = np.clip((x - A["NMU"]) / A["NSD"], -5, 5)
    prob = 0.0
    for mm in range(N_NET):
        HA = [max(A["B1"][mm * H1 + jj] + sum(A["W1"][mm * H1 * N_IN + jj * N_IN + ii] * Z[ii] for ii in range(N_IN)), 0.0)
              for jj in range(H1)]
        HB = [max(A["B2"][mm * H2 + jj] + sum(A["W2"][mm * H2 * H1 + jj * H1 + ii] * HA[ii] for ii in range(H1)), 0.0)
              for jj in range(H2)]
        acc = A["B3"][mm] + sum(A["W3"][mm * H2 + ii] * HB[ii] for ii in range(H2))
        prob += 1 / (1 + np.exp(-acc))
    return prob / N_NET


def verify_model(text, model, data, year, n_check=300):
    A = parse_arrays(text)
    rng = np.random.default_rng(0)
    errs = []
    for code in sorted(data):
        df = data[code]
        X, _ = core.feature_matrix(core.indicator_frame(df))
        rows = np.flatnonzero((df.index.year == year) & ~np.isnan(X).any(axis=1))
        if len(rows) == 0:
            continue
        for r in rng.choice(rows, size=min(6, len(rows)), replace=False):
            ref = core.predict_ensemble(model["models"], core.standardize_apply(X[r : r + 1], model["mu"], model["sd"]))[0]
            errs.append(abs(pl_forward(A, X[r].astype(np.float64)) - ref))
        if len(errs) >= n_check:
            break
    return max(errs), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="walk-forward year whose model to embed (default: latest)")
    args = ap.parse_args()
    data = load_all()
    codes = sorted(data)[::10]
    worst, worst_rel, nbars, base_diff = verify_features(data, codes)
    print(f"feature port: PowerLanguage replica vs core.feature_matrix on {len(codes)} stocks / {nbars} bars: "
          f"max |diff| = {worst:.2e} (relative {worst_rel:.2e}); base-rule bars differing = {base_diff}")
    assert worst_rel < 1e-6 and base_diff == 0, "PowerLanguage feature / rule port does not match Python"
    _, models = core.walk_forward(data)
    year = args.year or max(models)
    model = models[year]
    OUT.mkdir(exist_ok=True)
    for sid in ("kdj_macd_dl_floor", "kdj_macd_dl_gate"):
        text = build_dl(sid, year, model)
        (OUT / f"{sid}.txt").write_text(text, encoding="utf-8")
        err, n = verify_model(text, model, data, year)
        print(f"{sid}: embedded {year} model (n_train={model['n_train']}, QLO={model['q_lo']:.4f}, QHI={model['q_hi']:.4f}); "
              f"PowerLanguage replica vs Python max |dp| = {err:.2e} over {n} bars")
        assert err < 1e-5, "generated PowerLanguage weights do not reproduce the Python model"
    (OUT / "kdj_macd_dl_base.txt").write_text(build_base(), encoding="utf-8")
    print("kdj_macd_dl_base: written")


if __name__ == "__main__":
    main()
