"""Run the full study and write data/research/report.md + results.json (+ calibration.json for the live desk).

calibration.json is only written for a model that showed out-of-sample skill (positive Brier skill pooled AND in
most test years). Otherwise the desk gets an honest base rate and is told the history shows no reliable edge.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import backtest as bt
from . import features as ft
from . import sources

log = logging.getLogger("desk.research")
OUT = Path(__file__).resolve().parents[2] / "data" / "research"


def coverage(con: sqlite3.Connection) -> dict:
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    return {
        "bars": q("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM bars GROUP BY symbol"),
        "news_per_symbol": q("SELECT symbol, COUNT(*) FROM news_symbol GROUP BY symbol"),
        "news_range": q("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM news")[0],
        "news_scored": q("SELECT COUNT(*) FROM news_sentiment")[0][0],
        "macro": q("SELECT series, COUNT(*), MIN(date), MAX(date) FROM macro GROUP BY series"),
        "earnings": q("SELECT symbol, COUNT(*), MIN(filing_date), MAX(filing_date) FROM earnings GROUP BY symbol"),
    }


MIN_BRIER_SKILL = 0.002   # a meaningful improvement over the base rate, not a rounding error
MIN_YEARS_POSITIVE = 0.75
MIN_AUC = 0.52


def _bss(g: pd.DataFrame, model: str) -> float:
    base = brier_score_loss(g.y, g.p_base_rate)
    return 1 - brier_score_loss(g.y, g[f"p_{model}"]) / base if base else 0.0


def has_skill(pred: pd.DataFrame, model: str) -> tuple[bool, dict]:
    """Skill is judged on all test years except the last ("selection years"). The last test year is a holdout:
    it is never used to pick a model, and the chosen model must also beat the base rate there before it is
    exported. This guards against picking the luckiest of several models (winner's curse)."""
    years = sorted(pred.year.unique())
    if len(years) < 2:  # need at least one selection year and one holdout year
        return False, {"reason": "fewer than two test years"}
    sel, hold = years[:-1], years[-1]
    ps = pred[pred.year.isin(sel)]
    yearly = [_bss(pred[pred.year == y], model) for y in sel]
    share = float(np.mean([b > 0 for b in yearly])) if yearly else 0.0
    auc = roc_auc_score(ps.y, ps[f"p_{model}"]) if ps.y.nunique() > 1 and ps[f"p_{model}"].nunique() > 1 else 0.5
    bss = _bss(ps, model)
    ok = bss > MIN_BRIER_SKILL and share >= MIN_YEARS_POSITIVE and auc > MIN_AUC
    return bool(ok), {"selection_years": [int(y) for y in sel], "holdout_year": int(hold),
                      "selection_brier_skill": float(bss), "selection_auc": float(auc),
                      "years_positive_share": share,
                      "holdout_brier_skill": float(_bss(pred[pred.year == hold], model))}


def export_logit(f: pd.DataFrame, pred: pd.DataFrame, model: str) -> dict:
    """Fit the chosen logistic model on ALL history and export it as plain numbers (no sklearn at runtime),
    plus an isotonic map from raw probability to the hit rate actually observed out of sample."""
    kind, cols = bt.MODEL_SPECS[model]
    d = bt.prepare(f)
    cols = [c for c in cols if d[c].notna().mean() > 0.2]
    pipe = bt.make_model(kind).fit(d[cols], d.y)
    imp, sc, lr = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    ind = imp.indicator_.features_.tolist() if imp.indicator_ is not None else []
    # calibration map learned from selection-year out-of-sample predictions only (the holdout stays untouched)
    sel = pred[pred.year < pred.year.max()]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(sel[f"p_{model}"], sel.y)
    grid = np.linspace(0.3, 0.7, 41)
    return {
        "model": model, "features": cols, "medians": imp.statistics_.tolist(), "missing_indicators": ind,
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "calibration": {"raw": grid.round(3).tolist(), "observed": iso.predict(grid).round(4).tolist()},
        "horizon_days": ft.HORIZON, "target": "beats SPY over the next 10 trading days (next open to open)",
    }


SCORE_EDGES = (-10, 0, 1, 2, 3, 4, 10)
VIX_EDGES = (0, 13, 16, 20, 25, 30, 100)


def spy_by_vix(f: pd.DataFrame) -> list[dict]:
    """SPY's next-10-trading-day return grouped by the (lagged) VIX level on the decision day. Descriptive:
    daily rows overlap, so n overstates the number of independent observations by about 10x."""
    m = f[(f.symbol == ft.BENCH) & f.fwd_ret.notna() & f.vix.notna()]
    b = pd.cut(m.vix, bins=list(VIX_EDGES))
    t = m.groupby(b, observed=True).agg(n=("fwd_ret", "size"), avg=("fwd_ret", "mean"),
                                         up=("fwd_ret", lambda x: float((x > 0).mean())))
    return [{"lo": float(iv.left), "hi": float(iv.right), "n": int(r.n), "spy_avg_10d": float(r.avg),
             "spy_up_share": float(r.up)} for iv, r in t.iterrows()]


def base_rates(f: pd.DataFrame) -> dict:
    """Descriptive history the live desk can quote: how often stocks beat SPY over 10 days, overall, per symbol
    and per Scout screen-score bucket. Not a prediction model; just what happened, with sample sizes."""
    d = bt.prepare(f)
    b = pd.cut(d.screen_score, bins=list(SCORE_EDGES))
    by_score = d.groupby(b, observed=True).agg(n=("y", "size"), beat_spy=("y", "mean"),
                                               avg_excess=("excess_fwd", "mean"))
    by_sym = d.groupby("symbol").agg(n=("y", "size"), beat_spy=("y", "mean"), avg_excess=("excess_fwd", "mean"))
    return {
        "horizon_days": ft.HORIZON, "period": [d.date.min().date().isoformat(), d.date.max().date().isoformat()],
        "overall": {"n": int(len(d)), "beat_spy": float(d.y.mean()), "avg_excess": float(d.excess_fwd.mean())},
        "score_edges": list(SCORE_EDGES),
        "by_screen_score": [{"lo": float(iv.left), "hi": float(iv.right), "n": int(r.n), "beat_spy": float(r.beat_spy),
                             "avg_excess": float(r.avg_excess)} for iv, r in by_score.iterrows()],
        "by_symbol": {sym: {"n": int(r.n), "beat_spy": float(r.beat_spy), "avg_excess": float(r.avg_excess)}
                      for sym, r in by_sym.iterrows()},
    }


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else
                         ("" if pd.isna(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(con: sqlite3.Connection, out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log.info("building point-in-time features")
    f = ft.build(con)
    f.to_pickle(out / "features.pkl")
    log.info("features: %d rows, %d symbols, %s..%s", len(f), f.symbol.nunique(), f.date.min().date(),
             f.date.max().date())

    log.info("walk-forward fitting")
    pred = bt.walk_forward(f)
    pred.to_pickle(out / "predictions.pkl")
    scores = bt.score_models(pred)

    sims = []
    for m in bt.MODEL_SPECS:
        for thr in (0.0, 0.55, 0.60):
            r = bt.simulate(pred, m, top_k=3, threshold=thr)
            sims.append({k: v for k, v in r.__dict__.items() if k != "curve"})
    sims = pd.DataFrame(sims)

    log.info("signal and source audits")
    ic = bt.signal_ic(f, [c for c in bt.PRICE_FEATURES + bt.NEWS_FEATURES + bt.EARN_FEATURES])
    timing = bt.market_timing_ic(f)
    arts = sources.articles(con, f)
    validity = sources.finbert_validity(arts) if not arts.empty else pd.DataFrame()
    reliab = sources.source_reliability(arts) if not arts.empty else pd.DataFrame()

    skill = {m: has_skill(pred, m) for m in bt.MODEL_SPECS if m != "base_rate"}
    skilled = [m for m, (ok, _) in skill.items() if ok and bt.MODEL_SPECS[m][0] == "logit"]
    best = max(skilled, key=lambda m: skill[m][1]["selection_brier_skill"]) if skilled else None
    if best and skill[best][1]["holdout_brier_skill"] <= 0:
        log.info("%s passed on selection years but failed the holdout year; not exported", best)
        best = None
    calib_path = out / "calibration.json"
    if best:
        cal = export_logit(f, pred, best)
        cal["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cal["evidence"] = skill[best][1]
        calib_path.write_text(json.dumps(cal, indent=1))
    elif calib_path.exists():
        calib_path.unlink()  # never leave a stale model the evidence no longer supports

    rates = base_rates(f)
    rates["spy_by_vix"] = spy_by_vix(f)
    rates["caveats"] = [
        "Per-symbol rates are survivorship-biased: today's watchlist is made of past winners.",
        ("A stock-picking model showed out-of-sample skill (calibration.json); these tables are still descriptive."
         if best else "No tested stock-picking model or signal beat the base rate out of sample (see report.md); "
                      "these are history, not forecasts."),
    ]
    vt = timing[timing.signal == "vix"] if not timing.empty else timing
    if not vt.empty:
        r0 = vt.iloc[0]
        sig = abs(r0.t_stat) > 2.26 and r0.years_same_sign >= 0.7
        rates["vix_timing"] = {
            "mean_yearly_ic": float(r0.mean_yearly_ic), "t_stat": float(r0.t_stat),
            "years_same_sign": float(r0.years_same_sign), "significant": bool(sig),
            "summary": (f"{rates['period'][0][:4]}-{rates['period'][1][:4]}: a higher VIX was followed by "
                        f"{'better' if r0.mean_yearly_ic > 0 else 'worse'} SPY returns over the next 10 days "
                        f"(t {r0.t_stat:+.1f}, same direction in {r0.years_same_sign:.0%} of years)"
                        + ("" if sig else "; not statistically reliable") + "."),
        }
    rates["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "base_rates.json").write_text(json.dumps(rates, indent=1))
    base_rate = float(pred.y.mean())
    highlights = []
    t_crit = 2.26  # ~5% two-sided with ~9 degrees of freedom (yearly means); still not corrected for many signals
    sc = ic[ic.signal == "screen_score"]
    if not sc.empty:
        r0 = sc.iloc[0]
        highlights.append(f"Scout screen score: IC {r0.mean_ic:+.3f} (t {r0.t_stat:+.1f}) - "
                         + ("it has predicted relative returns." if abs(r0.t_stat) > t_crit else
                            "no reliable predictive value."))
    strong = ic[(ic.t_stat.abs() > t_crit) & (ic.years_same_sign >= 0.7)]
    for _, r0 in strong.head(2).iterrows():
        highlights.append(f"{r0.signal}: IC {r0.mean_ic:+.3f}, t {r0.t_stat:+.1f}, same sign in "
                          f"{r0.years_same_sign:.0%} of years.")
    vt = timing[timing.signal == "vix"] if not timing.empty else timing
    if not vt.empty:
        r0 = vt.iloc[0]
        highlights.append(f"VIX vs next-10-day SPY return: IC {r0.mean_yearly_ic:+.3f} (t {r0.t_stat:+.1f}) - "
                          + ("higher fear has been followed by better market returns." if r0.t_stat > t_crit else
                             "no reliable link." if abs(r0.t_stat) <= t_crit else "higher fear preceded weaker returns."))
    if not validity.empty:
        a = validity[validity.year == "all"]
        if not a.empty and pd.notna(a.agreement.iloc[0]):
            highlights.append(f"FinBERT read clear headlines the same way the market moved "
                              f"{a.agreement.iloc[0]:.0%} of the time (50% = coin flip).")
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": coverage(con), "rows": len(f), "test_rows": len(pred),
        "test_years": sorted(int(y) for y in pred.year.unique()),
        "base_rate_beat_spy": base_rate, "skill": skill, "chosen_model": best, "highlights": highlights,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))

    pooled = scores[scores.year == "all"].drop(columns="year").sort_values("brier_skill", ascending=False)
    by_year = scores[scores.year != "all"].pivot(index="model", columns="year", values="brier_skill")
    calib = bt.calibration_table(pred, best or "all")
    md = [
        "# Agent Desk: 10-year historical study", "",
        f"Generated {results['generated_at']}. Test years {results['test_years'][0]}–{results['test_years'][-1]} "
        f"(walk-forward, {bt.EMBARGO}-day embargo, {bt.COST:.2%} round-trip cost). "
        f"Target: does the stock beat SPY over the next {ft.HORIZON} trading days.", "",
        f"Base rate (share of stock-days that beat SPY): **{base_rate:.1%}**.", "",
        "## Models, pooled out-of-sample", _md_table(pooled), "",
        "Brier skill > 0 means better calibrated predictions than the base rate. AUC 0.5 = no ranking skill.", "",
        "## Brier skill by test year", _md_table(by_year.reset_index()), "",
        "## Trading simulation (top 3 picks every 10 days, vs holding SPY)",
        _md_table(sims[["model", "threshold", "invested_periods", "periods", "cagr", "universe_cagr", "spy_cagr",
                        "sharpe", "universe_sharpe", "spy_sharpe", "max_dd", "spy_max_dd", "beat_universe_share",
                        "avg_trade_excess"]]), "",
        "universe = equal weight in all watchlist stocks. Beating it is what stock-picking skill means; beating SPY "
        "alone mostly reflects which stocks are on today's watchlist (survivorship bias).", "",
        f"## Calibration ({best or 'all'}): predicted vs actual hit rate", _md_table(calib), "",
        "## Signals: information coefficient vs next-10-day excess return (per-year means)",
        _md_table(ic[["signal", "mean_ic", "t_stat", "years", "years_same_sign"]]), "",
        "|t| > 2 and the same sign in most years is the bar for a signal being real rather than noise.", "",
        "## Macro signals: do they predict the market (SPY next 10 days)?", _md_table(timing), "",
        "## FinBERT validity: clear calls that agreed with the day's move vs SPY", _md_table(validity), "",
        "## News source reliability", _md_table(reliab) if not reliab.empty else "_no scored news_", "",
        "## Base rates by Scout screen score (all history, descriptive)",
        _md_table(pd.DataFrame(rates["by_screen_score"])), "",
        "## Verdict", "",
        (f"**{best}** showed out-of-sample skill and was exported to calibration.json for the desk."
         if best else "**No model showed reliable out-of-sample skill over the base rate.** The desk will be "
         "given the honest base rate instead of a fitted model."), "",
    ]
    (out / "report.md").write_text("\n".join(md))
    log.info("wrote %s", out / "report.md")
    return results
