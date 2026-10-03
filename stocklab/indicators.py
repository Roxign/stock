"""Indicators matching common Taiwan charting conventions (e.g. 9,3,3 KDJ with 1/3 smoothing)."""

import numpy as np
import pandas as pd

# Moving averages Taiwanese traders watch: 週線, 雙週線, 月線, 季線, 半年線, 年線 (in trading days).
MA_PERIODS = {"week": 5, "biweek": 10, "month": 20, "quarter": 60, "half_year": 120, "year": 240}


def tw_kdj(df, n=9, k_smooth=3, d_smooth=3):
    """Taiwan-style KDJ: RSV over n bars, K = K[1]*(s-1)/s + RSV/s, D likewise, seeded at 50; J = 3K - 2D."""
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rng = (high_n - low_n).to_numpy()
    rsv = np.where(rng > 0, (df["close"].to_numpy() - low_n.to_numpy()) / np.where(rng > 0, rng, 1) * 100, 50.0)
    k = np.empty(len(df))
    d = np.empty(len(df))
    kp = dp = 50.0
    for i, r in enumerate(rsv):
        kp = kp * (k_smooth - 1) / k_smooth + r / k_smooth
        dp = dp * (d_smooth - 1) / d_smooth + kp / d_smooth
        k[i], d[i] = kp, dp
    return pd.DataFrame({"k": k, "d": d, "j": 3 * k - 2 * d}, index=df.index)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    """Returns DIF (fast EMA - slow EMA), MACD (signal EMA of DIF) and OSC (DIF - MACD)."""
    dif = ema(close, fast) - ema(close, slow)
    sig = ema(dif, signal)
    return pd.DataFrame({"dif": dif, "macd": sig, "osc": dif - sig}, index=close.index)


def sma(s, n):
    return s.rolling(n).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(df, n=14):
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd}, index=close.index)
