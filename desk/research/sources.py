"""Source audits: how reliable is each news source, and how accurate is the sentiment model itself?

Two different questions, measured separately:
1. Validity of the reader (FinBERT): when it calls a headline clearly positive/negative, did the stock's move
   relative to SPY on that headline's day go the same way? 50% = coin flip. This checks the tool, not the news.
2. Predictive value of each source: after a source's positive vs negative stories, did the stock beat SPY over the
   NEXT 10 trading days (entry at the next open, so the day's own reaction is excluded)? Measured per year, so a
   source that only "worked" in one lucky year shows up as such.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import BENCH, _calendar, load_bars, to_decision_day

STRONG = 0.5  # |p_pos - p_neg| above this counts as a clear call


def articles(con: sqlite3.Connection, f: pd.DataFrame, model: str = "ProsusAI/finbert") -> pd.DataFrame:
    q = """SELECT n.id, n.created_at, COALESCE(NULLIF(n.source, ''), 'unknown') AS source, ns.symbol, s.score
           FROM news n JOIN news_symbol ns ON ns.id = n.id
           JOIN news_sentiment s ON s.id = n.id AND s.model = ?"""
    a = pd.read_sql(q, con, params=(model,))
    if a.empty:
        return a
    bars = load_bars(con)
    cal = _calendar(bars)
    a["date"] = to_decision_day(a.created_at, cal)
    a = a[~a.symbol.isin({BENCH, "QQQ"})].dropna(subset=["date"])
    cols = f[["symbol", "date", "ret_1d", "excess_fwd"]].copy()
    spy = f[f.symbol == BENCH][["date", "ret_1d"]].rename(columns={"ret_1d": "spy_ret_1d"})
    cols = cols.merge(spy, on="date", how="left")
    cols["react_excess"] = cols.ret_1d - cols.spy_ret_1d  # the day's move vs SPY (includes the news itself)
    a = a.merge(cols, on=["symbol", "date"], how="inner")
    a["year"] = a.date.dt.year
    return a


def finbert_validity(a: pd.DataFrame) -> pd.DataFrame:
    s = a[(a.score.abs() > STRONG) & a.react_excess.notna() & (a.react_excess != 0)]
    agree = (np.sign(s.score) == np.sign(s.react_excess)).astype(float)
    t = s.assign(agree=agree).groupby("year").agg(n=("agree", "size"), agreement=("agree", "mean"))
    t.loc["all"] = [len(s), agree.mean() if len(s) else np.nan]
    return t.reset_index()


def source_reliability(a: pd.DataFrame, min_articles: int = 300) -> pd.DataFrame:
    rows = []
    for src, g in a.groupby("source"):
        g = g.dropna(subset=["excess_fwd"])
        if len(g) < min_articles:
            continue
        strong = g[g.score.abs() > STRONG]
        yearly = {}
        for yr, h in strong.groupby("year"):
            if len(h) >= 30 and h.score.nunique() > 2:
                yearly[yr] = spearmanr(h.score, h.excess_fwd).statistic
        ys = pd.Series(yearly, dtype=float)
        n = len(ys)
        t = ys.mean() / (ys.std(ddof=1) / math.sqrt(n)) if n > 2 and ys.std(ddof=1) > 0 else 0.0
        pos, neg = strong[strong.score > 0], strong[strong.score < 0]
        same_day = strong[strong.react_excess.notna() & (strong.react_excess != 0)]
        rows.append({
            "source": src, "articles": len(g), "clear_calls": len(strong),
            "first": g.date.min().date().isoformat(), "last": g.date.max().date().isoformat(),
            "same_day_agreement": float((np.sign(same_day.score) == np.sign(same_day.react_excess)).mean())
            if len(same_day) else np.nan,
            "excess_after_positive": pos.excess_fwd.mean(), "excess_after_negative": neg.excess_fwd.mean(),
            "pos_minus_neg": pos.excess_fwd.mean() - neg.excess_fwd.mean(),
            "mean_ic": ys.mean() if n else np.nan, "t_stat": t, "years": n,
            "years_same_sign": float((np.sign(ys) == np.sign(ys.mean())).mean()) if n else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["source", "articles"])
    return pd.DataFrame(rows).sort_values("articles", ascending=False)
