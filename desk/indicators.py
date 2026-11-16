"""Pure indicator functions over plain lists of floats."""
from __future__ import annotations

from typing import Optional, Sequence

from .models import Bar


def sma(values: Sequence[float], n: int) -> Optional[float]:
    if n <= 0 or len(values) < n:
        return None
    return sum(values[-n:]) / n


def rsi(closes: Sequence[float], n: int = 14) -> Optional[float]:
    """Wilder RSI. Needs n+1 closes."""
    if len(closes) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain, avg_loss = gains / n, losses / n
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (n - 1) + max(d, 0.0)) / n
        avg_loss = (avg_loss * (n - 1) + max(-d, 0.0)) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> Optional[float]:
    """Wilder ATR. Needs n+1 bars."""
    if len(closes) < n + 1 or not (len(highs) == len(lows) == len(closes)):
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    a = sum(trs[:n]) / n
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def volume_ratio(volumes: Sequence[float], n: int = 20, last_fraction: Optional[float] = 1.0) -> Optional[float]:
    """Latest volume divided by the average of the previous n volumes.

    `last_fraction` is the share of a normal day's volume that should have traded by now when the latest bar is
    today's still-forming bar (see session_volume_fraction). The latest volume is scaled up by it so a half-finished
    day isn't read as "volume is half of normal". None means it is too early in the session to judge.
    """
    if last_fraction is None or last_fraction <= 0 or len(volumes) < n + 1:
        return None
    avg = sum(volumes[-n - 1:-1]) / n
    return (volumes[-1] / min(last_fraction, 1.0)) / avg if avg > 0 else None


# Approximate cumulative share of a regular US session's volume traded by each minute after the 9:30 ET open.
# Volume is U-shaped (heavy at the open and into the close), so a straight minutes/390 would overstate the
# ratio in the morning and understate it in the afternoon.
_CUM_VOLUME = ((0, 0.0), (30, 0.13), (60, 0.21), (90, 0.28), (120, 0.34), (150, 0.40), (180, 0.45),
               (210, 0.51), (240, 0.56), (270, 0.62), (300, 0.68), (330, 0.75), (360, 0.84), (390, 1.0))
MIN_SESSION_MINUTES = 15  # before this, a partial day's volume says too little to compare


def session_volume_fraction(minutes_since_open: float) -> Optional[float]:
    """Expected share of a full day's volume traded `minutes_since_open` into the regular session (None if too early)."""
    if minutes_since_open < MIN_SESSION_MINUTES:
        return None
    if minutes_since_open >= _CUM_VOLUME[-1][0]:
        return 1.0
    for (m0, f0), (m1, f1) in zip(_CUM_VOLUME, _CUM_VOLUME[1:]):
        if m0 <= minutes_since_open <= m1:
            return f0 + (f1 - f0) * (minutes_since_open - m0) / (m1 - m0)
    return 1.0


def pct_return(closes: Sequence[float], n: int) -> Optional[float]:
    if len(closes) < n + 1 or closes[-n - 1] == 0:
        return None
    return closes[-1] / closes[-n - 1] - 1.0


def dist_from_high(closes: Sequence[float], highs: Sequence[float], n: int = 20) -> Optional[float]:
    """(close - n-day high) / n-day high; 0 means at the high, negative below it."""
    if len(highs) < n:
        return None
    hh = max(highs[-n:])
    return closes[-1] / hh - 1.0 if hh > 0 else None


def _r(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(x, d)


def compute_indicators(bars: Sequence[Bar], last_bar_fraction: Optional[float] = 1.0) -> dict:
    """Indicators from daily bars. `last_bar_fraction` < 1 marks the latest bar as today's partial bar."""
    c = [b.close for b in bars]
    h = [b.high for b in bars]
    lo = [b.low for b in bars]
    v = [b.volume for b in bars]
    return {
        "sma20": _r(sma(c, 20), 2),
        "sma50": _r(sma(c, 50), 2),
        "rsi14": _r(rsi(c, 14), 1),
        "atr14": _r(atr(h, lo, c, 14), 3),
        "volume_ratio20": _r(volume_ratio(v, 20, last_bar_fraction), 2),
        "ret_5d": _r(pct_return(c, 5)),
        "ret_20d": _r(pct_return(c, 20)),
        "dist_from_20d_high": _r(dist_from_high(c, h, 20)),
    }
