"""Walk-forward test: does any signal predict which watchlist stock beats SPY over the next 10 trading days?

Honesty rules:
- Every prediction for year Y comes from a model fitted only on rows whose label window ended before Y began
  (an embargo of HORIZON+5 trading days separates train and test, because 10-day labels overlap).
- The target is "beats SPY", not "goes up": in a rising market almost everything goes up, and a model that learns
  that is only learning to hold the index.
- Trading results are net of a round-trip cost and compared against simply holding SPY over the same periods.
- The comparison that matters is against the base rate (no skill). A model "has skill" only if it beats that
  out-of-sample, in most years, not just pooled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_GROUPS, BENCH, HORIZON, MACRO_FEATURES, NEWS_FEATURES, PRICE_FEATURES, EARN_FEATURES

COST = 0.001            # 10 bps round trip (spread + slippage; Alpaca charges no commission)
EMBARGO = HORIZON + 5   # trading days between the last training label and the first test row
MIN_TRAIN_YEARS = 3

MODEL_SPECS: dict[str, tuple[str, list[str]]] = {
    "base_rate": ("const", []),
    "scout_rule": ("logit", ["screen_score"]),
    "price": ("logit", PRICE_FEATURES),
    "price+macro": ("logit", PRICE_FEATURES + MACRO_FEATURES),
    "price+news": ("logit", PRICE_FEATURES + NEWS_FEATURES),
    "price+earnings": ("logit", PRICE_FEATURES + EARN_FEATURES),
    "all": ("logit", PRICE_FEATURES + MACRO_FEATURES + NEWS_FEATURES + EARN_FEATURES),
    "all_gbm": ("gbm", PRICE_FEATURES + MACRO_FEATURES + NEWS_FEATURES + EARN_FEATURES),
}


def make_model(kind: str):
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=300,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


NOT_STOCKS = {BENCH, "QQQ"}  # index ETFs are benchmarks, not stock picks


def prepare(f: pd.DataFrame) -> pd.DataFrame:
    d = f[~f.symbol.isin(NOT_STOCKS) & f.excess_fwd.notna()].copy()
    d["y"] = (d.excess_fwd > 0).astype(int)
    d["year"] = d.date.dt.year
    return d


def walk_forward(f: pd.DataFrame, specs: dict = MODEL_SPECS) -> pd.DataFrame:
    """Out-of-sample probability from every model for every row in every test year."""
    d = prepare(f)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    years = sorted(d.year.unique())
    first_test = years[0] + MIN_TRAIN_YEARS
    out = []
    for y in [yy for yy in years if yy >= first_test]:
        test = d[d.year == y]
        start_pos = dates.searchsorted(pd.Timestamp(y, 1, 1))
        cutoff = dates[max(0, start_pos - EMBARGO)]
        train = d[d.date < cutoff]
        res = test[["symbol", "date", "year", "y", "fwd_ret", "excess_fwd", "spy_fwd_ret"]].copy()
        for name, (kind, cols) in specs.items():
            if kind == "const":
                res[f"p_{name}"] = train.y.mean()
                continue
            usable = [c for c in cols if train[c].notna().mean() > 0.2]  # skip groups with no data yet
            if not usable:
                res[f"p_{name}"] = train.y.mean()
                continue
            m = make_model(kind).fit(train[usable], train.y)
            res[f"p_{name}"] = m.predict_proba(test[usable])[:, 1]
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ---------- scoring ----------

def score_models(pred: pd.DataFrame) -> pd.DataFrame:
    """Per model and year: AUC, Brier, Brier skill vs base rate, and spread of top vs bottom picks."""
    rows = []
    models = [c[2:] for c in pred.columns if c.startswith("p_")]
    for (yr, g) in list(pred.groupby("year")) + [("all", pred)]:
        base_brier = brier_score_loss(g.y, g.p_base_rate)
        for m in models:
            p = g[f"p_{m}"]
            auc = roc_auc_score(g.y, p) if g.y.nunique() > 1 and p.nunique() > 1 else 0.5
            brier = brier_score_loss(g.y, p)
            # within each day: excess return of the model's top pick minus the average stock (random tie-break,
            # so a constant model doesn't always "pick" the alphabetically first symbol)
            tb = g.assign(_tb=np.random.default_rng(0).random(len(g)))
            top = tb.sort_values([f"p_{m}", "_tb"], ascending=False).groupby("date").head(1)
            spread = top.excess_fwd.mean() - g.excess_fwd.mean()
            rows.append({"year": yr, "model": m, "n": len(g), "auc": auc, "brier": brier,
                         "brier_skill": 1 - brier / base_brier if base_brier else 0.0,
                         "top_pick_edge": spread})
    return pd.DataFrame(rows)


def calibration_table(pred: pd.DataFrame, model: str, edges=(0, .40, .45, .50, .55, .60, .65, 1.0)) -> pd.DataFrame:
    p = pred[f"p_{model}"]
    b = pd.cut(p, bins=list(edges), include_lowest=True)
    t = pred.groupby(b, observed=True).agg(n=("y", "size"), predicted=(f"p_{model}", "mean"), actual=("y", "mean"),
                                            avg_excess=("excess_fwd", "mean"))
    return t.reset_index().rename(columns={f"p_{model}": "bucket"})


@dataclass
class SimResult:
    model: str
    threshold: float
    periods: int
    invested_periods: int
    cagr: float
    spy_cagr: float
    sharpe: float
    spy_sharpe: float
    max_dd: float
    spy_max_dd: float
    universe_cagr: float      # equal weight in every watchlist stock: the honest benchmark for stock picking
    universe_sharpe: float
    beat_universe_share: float
    beat_spy_share: float
    avg_trade_excess: float
    curve: pd.DataFrame = field(repr=False, default=None)


def _stats(r: pd.Series, per_year: float) -> tuple[float, float, float]:
    eq = (1 + r).cumprod()
    yrs = len(r) / per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else float("nan")
    sd = r.std()
    sharpe = r.mean() / sd * math.sqrt(per_year) if sd > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return cagr, sharpe, dd


def simulate(pred: pd.DataFrame, model: str, top_k: int = 3, threshold: float = 0.0) -> SimResult:
    """Non-overlapping 10-day periods: buy the top_k stocks whose probability >= threshold, equal weight, hold
    HORIZON days (cash if nothing qualifies). Compared to holding SPY over the same periods."""
    dates = sorted(pred.date.unique())
    rebal = dates[::HORIZON]
    strat, spy, univ, trades = [], [], [], []
    rng = np.random.default_rng(0)
    for d in rebal:
        g = pred[pred.date == d]
        if g.empty:
            continue
        # random tie-break, so a constant model picks at random rather than alphabetically
        g = g.assign(_tb=rng.random(len(g))).sort_values([f"p_{model}", "_tb"], ascending=False)
        pick = g[g[f"p_{model}"] >= threshold].head(top_k)
        spy.append(g.spy_fwd_ret.iloc[0])
        univ.append(g.fwd_ret.mean() - COST)
        if pick.empty:
            strat.append(0.0)
        else:
            strat.append(pick.fwd_ret.mean() - COST)
            trades.extend((pick.excess_fwd - COST).tolist())
    idx = rebal[:len(strat)]
    s, b, u = pd.Series(strat, index=idx), pd.Series(spy, index=idx), pd.Series(univ, index=idx)
    per_year = 252 / HORIZON
    c, sh, dd = _stats(s, per_year)
    bc, bsh, bdd = _stats(b, per_year)
    uc, ush, _ = _stats(u, per_year)
    curve = pd.DataFrame({"strategy": (1 + s).cumprod(), "spy": (1 + b).cumprod(), "universe": (1 + u).cumprod()})
    return SimResult(model, threshold, len(s), int((s != 0).sum()), c, bc, sh, bsh, dd, bdd, uc, ush,
                     float((s > u).mean()), float((s > b).mean()), float(np.mean(trades)) if trades else float("nan"), curve)


# ---------- signal and source audits ----------

def signal_ic(f: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Information coefficient: daily cross-sectional Spearman correlation of a signal with next-10-day excess
    return, averaged per year. t-stat uses the yearly means (robust to overlapping labels)."""
    d = prepare(f)
    cols = cols or [c for grp in ALL_GROUPS.values() for c in grp]
    rows = []
    for c in cols:
        if c not in d or d[c].notna().mean() < 0.2:
            continue
        def day_ic(g):
            g = g[[c, "excess_fwd"]].dropna()
            if len(g) < 5 or g[c].nunique() < 3:
                return np.nan
            return spearmanr(g[c], g.excess_fwd).statistic
        daily = d.groupby("date").apply(day_ic, include_groups=False).dropna()
        yearly = daily.groupby(daily.index.year).mean()
        n = len(yearly)
        t = yearly.mean() / (yearly.std(ddof=1) / math.sqrt(n)) if n > 2 and yearly.std(ddof=1) > 0 else 0.0
        same_sign = float((np.sign(yearly) == np.sign(yearly.mean())).mean()) if n else 0.0
        rows.append({"signal": c, "mean_ic": yearly.mean(), "t_stat": t, "years": n,
                     "years_same_sign": same_sign, **{str(k): v for k, v in yearly.items()}})
    if not rows:
        return pd.DataFrame(columns=["signal", "mean_ic", "t_stat", "years", "years_same_sign"])
    return pd.DataFrame(rows).sort_values("t_stat", key=lambda s: -s.abs())


def market_timing_ic(f: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Macro signals are the same for every stock on a day, so test them on the market itself: does the signal
    predict SPY's next-10-day return? Non-overlapping 10-day samples; per-year Spearman; t-stat on yearly values."""
    cols = cols or MACRO_FEATURES
    m = f[f.symbol == BENCH].dropna(subset=["fwd_ret"]).sort_values("date").iloc[::HORIZON]
    rows = []
    for c in cols:
        if c not in m or m[c].notna().mean() < 0.5:
            continue
        g = m[["date", c, "fwd_ret"]].dropna()
        yearly = g.groupby(g.date.dt.year).apply(
            lambda h: spearmanr(h[c], h.fwd_ret).statistic if len(h) >= 10 else np.nan, include_groups=False).dropna()
        n = len(yearly)
        t = yearly.mean() / (yearly.std(ddof=1) / math.sqrt(n)) if n > 2 and yearly.std(ddof=1) > 0 else 0.0
        pooled = spearmanr(g[c], g.fwd_ret).statistic
        rows.append({"signal": c, "pooled_ic": pooled, "mean_yearly_ic": yearly.mean(), "t_stat": t, "years": n,
                     "years_same_sign": float((np.sign(yearly) == np.sign(yearly.mean())).mean()) if n else 0.0})
    if not rows:
        return pd.DataFrame(columns=["signal", "pooled_ic", "mean_yearly_ic", "t_stat", "years", "years_same_sign"])
    return pd.DataFrame(rows).sort_values("t_stat", key=lambda s: -s.abs())
