"""Generate MultiCharts PowerLanguage files for the kd_macd_dl strategies, embedding a frozen walk-forward model.

  python -m strategies.kd_macd_dl.gen_multicharts            (run from the repo root)
  python -m strategies.kd_macd_dl.gen_multicharts --year 2026

By default the model of the most recent walk-forward year is embedded (the same model the backtest uses for that
year: trained on every sample whose label ended before Jan 1 of that year). Re-run each January (after updating the
data) to retrain. After writing, the script parses the weights back out of the generated text and replays the
PowerLanguage forward pass (same flat-array index arithmetic, same 8-decimal constants) against the Python model.
"""

import argparse
import re
from pathlib import Path

import numpy as np

from stocklab.data import load_all

from . import core

HERE = Path(__file__).resolve().parent
OUT = HERE / "multicharts"
N_IN, H1, H2 = 28, core.HIDDEN[0], core.HIDDEN[1]
N_NET = len(core.SEEDS)


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


DAILY = """    { ---------- 台灣 KD(9,3,3)：與 stocklab.indicators.tw_kd 相同 ---------- }
    hh = Highest(High, 9);
    ll = Lowest(Low, 9);
    if hh - ll <> 0 then rsv = (Close - ll) / (hh - ll) * 100 else rsv = 50;
    kv = kv[1] * 2 / 3 + rsv / 3;
    dv = dv[1] * 2 / 3 + kv / 3;

    { ---------- MACD(12,26,9) ---------- }
    dif = XAverage(Close, 12) - XAverage(Close, 26);
    sigv = XAverage(dif, 9);
    osc = dif - sigv;
"""

SLOW_AND_RULE = """    { ---------- 週線等效 KD(45,15,15)：日 K 上把 9,3,3 乘以 5 ---------- }
    hhw = Highest(High, 45);
    llw = Lowest(Low, 45);
    if hhw - llw <> 0 then rsvw = (Close - llw) / (hhw - llw) * 100 else rsvw = 50;
    kw = kw[1] * 14 / 15 + rsvw / 15;
    dw = dw[1] * 14 / 15 + kw / 15;

    { ---------- 週線等效 MACD(60,130,45) ---------- }
    difw = XAverage(Close, 60) - XAverage(Close, 130);
    sigw = XAverage(difw, 45);
    oscw = difw - sigw;

    { ---------- 基礎規則：週線等效 KD 與 MACD 同向才換狀態 ---------- }
    if baseState = 0 and kw > dw and oscw > 0 then baseState = 1
    else if baseState = 1 and kw < dw and oscw < 0 then baseState = 0;
"""

MODEL = """    { ---------- ATR(14)，Wilder 平滑：與 stocklab.indicators.atr 相同 ---------- }
    if CurrentBar = 1 then atrv = High - Low
    else atrv = atrv[1] + (TrueRange - atrv[1]) / 14;

    { ---------- 28 個輸入特徵（順序必須與訓練時相同） ---------- }
    for ii = 0 to 4 begin
        Xr[2 * ii] = kv[ii] / 100 - 0.5;
        Xr[2 * ii + 1] = dv[ii] / 100 - 0.5;
        if atrv[ii] > 0 then begin
            Xr[10 + 2 * ii] = dif[ii] / atrv[ii];
            Xr[11 + 2 * ii] = osc[ii] / atrv[ii];
        end else begin
            Xr[10 + 2 * ii] = NMU[10 + 2 * ii];
            Xr[11 + 2 * ii] = NMU[11 + 2 * ii];
        end;
    end;
    for jj = 0 to 1 begin
        lagw = 5 * jj;
        Xr[20 + 4 * jj] = kw[lagw] / 100 - 0.5;
        Xr[21 + 4 * jj] = dw[lagw] / 100 - 0.5;
        if atrv[lagw] > 0 then begin
            Xr[22 + 4 * jj] = difw[lagw] / atrv[lagw];
            Xr[23 + 4 * jj] = oscw[lagw] / atrv[lagw];
        end else begin
            Xr[22 + 4 * jj] = NMU[22 + 4 * jj];
            Xr[23 + 4 * jj] = NMU[23 + 4 * jj];
        end;
    end;

    { ---------- 標準化（訓練集平均/標準差），截斷在 ±5 ---------- }
    for ii = 0 to 27 begin
        zz = (Xr[ii] - NMU[ii]) / NSD[ii];
        if zz > 5 then zz = 5;
        if zz < -5 then zz = -5;
        Z[ii] = zz;
    end;

    { ---------- 3 個 MLP（28-16-8-1, ReLU, sigmoid）取平均 ---------- }
    prob = 0;
    for mm = 0 to 2 begin
        for jj = 0 to 15 begin
            acc = B1[mm * 16 + jj];
            for ii = 0 to 27 begin
                acc = acc + W1[mm * 448 + jj * 28 + ii] * Z[ii];
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
"""

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
    "kd_macd_dl_floor": """    { ---------- 部位：底倉 0.5；KD+MACD 多頭 或 MLP 相對強 → 1.0 ---------- }
    if baseState = 1 or probS >= QHI then tgt = 1 else tgt = 0.5;
""",
    "kd_macd_dl_gate": """    { ---------- 部位：KD+MACD 多頭 或 MLP 相對強 → 1；KD+MACD 空頭 且 MLP 相對弱 → 0；其他 → 0.5 ---------- }
    if baseState = 1 or probS >= QHI then tgt = 1
    else if probS < QLO then tgt = 0
    else tgt = 0.5;
""",
}

TITLE = {
    "kd_macd_dl_floor": "KD+MACD+MLP 半倉底倉",
    "kd_macd_dl_gate": "KD+MACD+MLP 三段部位 (0 / 0.5 / 1)",
}

PYRAMID_NOTE = {
    True: """  2. 本策略會加碼與部分減碼，請在 Strategy Properties 勾選允許同方向多筆進場
     (Allow up to N entry orders in the same direction，N 設 2 以上)。""",
    False: "  2. 本策略只有全進全出，不需要允許同方向多筆進場。",
}


def common_header(pyramid):
    return """  使用前請注意：
  1. 請在 Strategy Properties 設定交易成本：買進手續費 0.1425%、賣出手續費 0.1425% + 證交稅 0.3%，
     以及合理的滑價；本程式碼本身不扣任何成本。
""" + PYRAMID_NOTE[pyramid] + """
  3. Input StartCapital 請設成與 Strategy Properties 的 Initial Capital 相同；LotSize = 1 表示可下零股，
     改成 1000 則只下整張。下單股數以訊號當根收盤價估算，Python 回測則以次日開盤價計算，會有些微差異。
  4. 請載入至少 2~3 年的日 K 資料：週線等效 MACD(60,130,45) 需要很長的暖機期，WarmBars 之前不下單。
  5. 訊號在 K 棒收盤時計算，於「下一根開盤」以市價成交，與網站上 Python 回測的規則一致。"""


def header_dl(sid, year, model):
    return f"""{{ ==========================================================================================
  {sid} — {TITLE[sid]}
  家族：KD+MACD+深度學習（由 strategies/kd_macd_dl/gen_multicharts.py 自動產生，請勿手動修改權重）

  【內嵌模型為凍結版本】
  MultiCharts 無法訓練模型。本檔內嵌的是 Python 前推式訓練中 {year} 年使用的模型：
  訓練樣本 = 50 檔成分股中，60 日標籤在 {year}-01-01 之前已結束的 {model['n_train']:,} 筆資料。
  模型不會自己更新；要用最新資料重新訓練，請在 Python 端更新資料後重新執行
      python -m strategies.kd_macd_dl.gen_multicharts
  再把新產生的程式碼貼回 MultiCharts 重新編譯。建議每年 1 月更新一次（與回測的重訓頻率相同）。

  模型：{N_NET} 個多層感知器 (MLP) {N_IN}→{H1}→{H2}→1，每個 {core.n_params(N_IN):,} 個參數，合計 {N_NET * core.n_params(N_IN):,} 個參數，輸出取平均。
  輸入：日 KD(9,3,3) 的 K、D（最近 5 根）、日 MACD(12,26,9) 的 DIF 與 OSC 除以 ATR(14)（最近 5 根）、
        週線等效 KD(45,15,15) 與 MACD(60,130,45)（今天與 5 根前）。
  輸出：未來 60 個交易日「贏過 0050 成分股中位數」的機率；平滑後與訓練集預測值的
        第 20 百分位 (QLO = {fmt(model['q_lo'])}) / 第 40 百分位 (QHI = {fmt(model['q_hi'])}) 比較。

{common_header(True)}
  ========================================================================================== }}
"""


def build_dl(sid, year, model):
    arrs = flatten(model)
    return f"""{header_dl(sid, year, model)}
Inputs:
    StartCapital(1000000),
    LotSize(1),
    WarmBars(250);

Variables:
    hh(0), ll(0), rsv(50), kv(50), dv(50),
    hhw(0), llw(0), rsvw(50), kw(50), dw(50),
    dif(0), sigv(0), osc(0), difw(0), sigw(0), oscw(0),
    atrv(0), baseState(0),
    ii(0), jj(0), mm(0), lagw(0), zz(0), acc(0), prob(0), probS(0),
    QLO({fmt(model['q_lo'])}), QHI({fmt(model['q_hi'])}),
    tgt(0), lastTgt(-1), eqv(0), wantSh(0), diffSh(0), sellSh(0);

Arrays:
    Xr[27](0), Z[27](0), HA[15](0), HB[7](0),
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
  kd_macd_dl_base — 週KD+MACD 基礎規則（無 DL）
  家族：KD+MACD+深度學習（DL 版本所依附的基礎規則，沒有任何模型，用來對照 DL 是否加分）

  規則：日線資料上使用「週線等效」參數 KD(45,15,15)、MACD(60,130,45)。
        K > D 且 OSC > 0 → 次日開盤全數買進；K < D 且 OSC < 0 → 次日開盤全數賣出；其餘維持原狀態。

{common_header(False)}
  ========================================================================================== }}

Inputs:
    StartCapital(1000000),
    LotSize(1),
    WarmBars(250);

Variables:
    hhw(0), llw(0), rsvw(50), kw(50), dw(50),
    difw(0), sigw(0), oscw(0),
    baseState(0), tgt(0), lastTgt(-1), eqv(0), wantSh(0), diffSh(0), sellSh(0);

{SLOW_AND_RULE}
    tgt = baseState;

{ORDERS}"""


# ---------------------------------------------------------------- verification of the generated code

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
        HA = [max(A["B1"][mm * 16 + jj] + sum(A["W1"][mm * 448 + jj * 28 + ii] * Z[ii] for ii in range(28)), 0.0) for jj in range(16)]
        HB = [max(A["B2"][mm * 8 + jj] + sum(A["W2"][mm * 128 + jj * 16 + ii] * HA[ii] for ii in range(16)), 0.0) for jj in range(8)]
        acc = A["B3"][mm] + sum(A["W3"][mm * 8 + ii] * HB[ii] for ii in range(8))
        prob += 1 / (1 + np.exp(-acc))
    return prob / N_NET


def verify(text, model, data, year, n_check=300):
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
    _, models = core.walk_forward(data)
    year = args.year or max(models)
    model = models[year]
    OUT.mkdir(exist_ok=True)
    for sid in ("kd_macd_dl_floor", "kd_macd_dl_gate"):
        text = build_dl(sid, year, model)
        (OUT / f"{sid}.txt").write_text(text, encoding="utf-8")
        err, n = verify(text, model, data, year)
        print(f"{sid}: embedded {year} model (n_train={model['n_train']}, QLO={model['q_lo']:.4f}, QHI={model['q_hi']:.4f}); "
              f"PowerLanguage replica vs Python max |dp| = {err:.2e} over {n} bars")
        assert err < 1e-5, "generated PowerLanguage weights do not reproduce the Python model"
    (OUT / "kd_macd_dl_base.txt").write_text(build_base(), encoding="utf-8")
    print("kd_macd_dl_base: written")


if __name__ == "__main__":
    main()
