"""Day-trading study: do classic intraday trend / indicator / news rules make money after costs?

Every rule is flat by the close (no overnight risk), uses only information available at decision time, and pays
a per-side cost (spread + slippage) on every entry, flip and exit. Variants are chosen on TRAIN years only and
judged on HOLDOUT years they never saw. Results: data/research/daytrade.json and daytrade.md.

    uv run --group research python -m desk.research.daytrade
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from desk.research.intraday import DEFAULT_PATH as INTRADAY_DB, ETFS, UNIVERSE

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DB = ROOT / "data" / "history.db"
OUT = ROOT / "data" / "research"
BARS = 78                      # 09:30 .. 15:55 five-minute bars
TRAIN = range(2018, 2023)      # 2018-2022: pick variants here
HOLDOUT = range(2023, 2027)    # 2023-2026: judged here, untouched while choosing
COST_BPS = {s: (1.0 if s in ("SPY", "QQQ", "IWM", "DIA") else 2.0) for s in ETFS}  # per side
STOCK_COST_BPS = 3.0


def cost_of(sym: str, mult: float = 1.0) -> float:
    return COST_BPS.get(sym, STOCK_COST_BPS) * mult / 1e4


# ------------------------------------------------------------------------------------------------ data

@dataclass
class Day:
    """One symbol's regular sessions as (days x 78) arrays, plus the previous close and pre-market move."""
    sym: str
    dates: np.ndarray       # datetime64[D]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    prev_close: np.ndarray  # previous session's 15:55 bar close (nan if unknown)


def load(sym: str, con: sqlite3.Connection) -> Day | None:
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM bars5 WHERE symbol=? AND ts >= ? "
                     "ORDER BY ts", con, params=(sym, "2018-01-01"))
    if df.empty:
        return None
    ts = pd.to_datetime(df.ts)
    df["date"], df["slot"] = ts.dt.normalize(), (ts.dt.hour * 60 + ts.dt.minute - 570) // 5
    df = df[(df.slot >= 0) & (df.slot < BARS)]
    counts = df.groupby("date").slot.count()
    full = counts[counts >= BARS - 6].index      # drop half-days and badly gapped days
    df = df[df.date.isin(full)]
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        m = df.pivot(index="date", columns="slot", values=col).reindex(columns=range(BARS))
        out[col] = m
    c = out["close"].ffill(axis=1).bfill(axis=1)
    o = out["open"].fillna(c.shift(1, axis=1)).fillna(c)
    h = out["high"].fillna(np.maximum(o, c))
    lo = out["low"].fillna(np.minimum(o, c))
    v = out["volume"].fillna(0.0)
    dates = c.index.values.astype("datetime64[D]")
    prev = np.r_[np.nan, c.values[:-1, -1]]
    gap_days = np.r_[999, np.diff(dates).astype(int)]
    prev[gap_days > 5] = np.nan                  # data hole: don't trust a "previous" close from weeks ago
    return Day(sym, dates, o.values, h.values, lo.values, c.values, v.values, prev)


def load_vix() -> pd.Series:
    """VIX close known before each session opens (previous trading day's close)."""
    if not HISTORY_DB.exists():
        return pd.Series(dtype=float)
    with sqlite3.connect(HISTORY_DB) as con:
        s = pd.read_sql("SELECT date, value FROM macro WHERE series='VIXCLS' ORDER BY date", con)
    s = s.set_index(pd.to_datetime(s.date)).value
    return s.shift(1)  # FRED stamps the close on its own date; use it only from the next session


def load_news(sym: str) -> pd.DataFrame:
    """Per session: FinBERT score of headlines published from the previous 16:00 to 09:25 New York."""
    if not HISTORY_DB.exists():
        return pd.DataFrame()
    q = ("SELECT n.created_at, s.score FROM news n JOIN news_symbol ns ON ns.id=n.id "
         "JOIN news_sentiment s ON s.id=n.id WHERE ns.symbol=?")
    with sqlite3.connect(HISTORY_DB) as con:
        df = pd.read_sql(q, con, params=(sym,))
    if df.empty:
        return df
    t = pd.to_datetime(df.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    # Headlines after 16:00 count for the next day; those before 09:25 count for today; skip intraday ones.
    mins = t.dt.hour * 60 + t.dt.minute
    day = t.dt.normalize()
    df["session"] = np.where(mins >= 960, day + pd.Timedelta(days=1), day)
    df = df[(mins >= 960) | (mins < 565)]
    df["session"] = pd.to_datetime(df.session)
    # weekend / holiday headlines roll to the next trading day when joined (asof forward below)
    return df.groupby("session").score.agg(["mean", "count"])


# ------------------------------------------------------------------------------------------ simulation

def run_path(d: Day, pos: np.ndarray, cost: float) -> np.ndarray:
    """pos[t, i] = position held from bar i's close to bar i+1's close (i = 0..76); flat at 15:55 close."""
    r = d.C[:, 1:] / d.C[:, :-1] - 1.0
    gross = np.nansum(pos[:, :-1] * r, axis=1)
    p = np.nan_to_num(pos[:, :-1])
    turns = np.abs(p[:, 0]) + np.abs(np.diff(p, axis=1)).sum(axis=1) + np.abs(p[:, -1])
    return gross - turns * cost


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out = 2 / (n + 1), np.empty_like(x)
    out[:, 0] = x[:, 0]
    for i in range(1, x.shape[1]):
        out[:, i] = a * x[:, i] + (1 - a) * out[:, i - 1]
    return out


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    dlt = np.diff(c, axis=1, prepend=c[:, :1])
    up, dn = np.clip(dlt, 0, None), np.clip(-dlt, 0, None)
    au, ad = ema(up, 2 * n - 1), ema(dn, 2 * n - 1)  # Wilder smoothing == EMA with span 2n-1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ad == 0, np.where(au == 0, 50.0, 100.0), 100 - 100 / (1 + au / ad))


def strat_intraday_momentum(d: Day, cost: float, vix_ok: np.ndarray, *, entry_slot: int = 72) -> np.ndarray:
    """Gao-Han-Li-Zhou (2018): the first half hour's move (prev close -> 10:00) predicts the last half hour."""
    first = d.C[:, 5] / d.prev_close - 1
    s = np.sign(np.nan_to_num(first)) * vix_ok
    ret = d.C[:, -1] / d.C[:, entry_slot - 1] - 1
    return s * ret - np.abs(s) * 2 * cost


def strat_orb(d: Day, cost: float, vix_ok: np.ndarray, *, minutes: int = 5, target_r: float = 0.0,
              long_only: bool = False, rvol_min: float = 0.0) -> np.ndarray:
    """Opening-range breakout: trade in the direction of the first N minutes, stop at the far side of the range,
    optional profit target at target_r x risk, otherwise exit at the 15:55 close. rvol_min > 0 keeps only
    "in play" days: opening-range volume at least rvol_min x its average over the previous 14 sessions."""
    k = minutes // 5
    out = np.zeros(len(d.dates))
    hi, lo = d.H[:, :k].max(axis=1), d.L[:, :k].min(axis=1)
    direction = np.sign(d.C[:, k - 1] - d.O[:, 0]) * vix_ok
    if rvol_min > 0:
        ov = pd.Series(d.V[:, :k].sum(axis=1))
        avg = ov.rolling(14, min_periods=10).mean().shift(1).values
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = direction * (np.nan_to_num(ov.values / avg) >= rvol_min)
    for t in np.nonzero(direction)[0]:
        s = direction[t]
        if long_only and s < 0:
            continue
        entry = d.O[t, k]
        stop = lo[t] if s > 0 else hi[t]
        risk = s * (entry - stop)
        if not np.isfinite(entry) or risk <= 0:   # opened beyond the range already: no trade
            continue
        target = entry + s * target_r * risk if target_r > 0 else None
        exit_px = d.C[t, -1]
        for i in range(k, BARS):  # stop checked before target inside a bar: the conservative assumption
            if s > 0 and d.L[t, i] <= stop:
                exit_px = min(d.O[t, i], stop)   # a gap through the stop fills at the worse open
                break
            if s < 0 and d.H[t, i] >= stop:
                exit_px = max(d.O[t, i], stop)
                break
            if target is not None and ((s > 0 and d.H[t, i] >= target) or (s < 0 and d.L[t, i] <= target)):
                exit_px = target
                break
        out[t] = s * (exit_px / entry - 1) - 2 * cost
    return out


def strat_vwap(d: Day, cost: float, vix_ok: np.ndarray, *, start_slot: int = 6, long_only: bool = False) -> np.ndarray:
    tp = (d.H + d.L + d.C) / 3
    cv = np.cumsum(d.V, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cv > 0, np.cumsum(tp * d.V, axis=1) / cv, d.C)
    pos = np.sign(d.C - vwap)
    if long_only:
        pos = np.clip(pos, 0, None)
    pos[:, :start_slot] = 0
    pos[:, -1] = 0
    return run_path(d, pos * vix_ok[:, None], cost)


def strat_ema_cross(d: Day, cost: float, vix_ok: np.ndarray, *, fast: int = 9, slow: int = 21,
                    start_slot: int = 6) -> np.ndarray:
    # Seed with the previous close so the averages are meaningful from the open.
    seed = np.where(np.isfinite(d.prev_close), d.prev_close, d.O[:, 0])[:, None]
    c = np.hstack([np.repeat(seed, slow, axis=1), d.C])
    pos = np.sign(ema(c, fast) - ema(c, slow))[:, slow:]
    pos[:, :start_slot] = 0
    pos[:, -1] = 0
    return run_path(d, pos * vix_ok[:, None], cost)


def strat_rsi_revert(d: Day, cost: float, vix_ok: np.ndarray, *, n: int = 14, lo: float = 30, hi: float = 70) -> np.ndarray:
    r = rsi(d.C, n)
    pos = np.zeros_like(d.C)
    cur = np.zeros(len(d.dates))
    for i in range(n, BARS - 1):
        cur = np.where(r[:, i] < lo, 1.0, np.where(r[:, i] > hi, -1.0, cur))
        cur = np.where((cur > 0) & (r[:, i] >= 50), 0.0, cur)
        cur = np.where((cur < 0) & (r[:, i] <= 50), 0.0, cur)
        pos[:, i] = cur
    return run_path(d, pos * vix_ok[:, None], cost)


def strat_gap(d: Day, cost: float, vix_ok: np.ndarray, *, min_gap: float = 0.005, fade: bool = True) -> np.ndarray:
    gap = d.O[:, 0] / d.prev_close - 1
    s = np.where(np.abs(np.nan_to_num(gap)) >= min_gap, np.sign(np.nan_to_num(gap)), 0.0)
    s = (-s if fade else s) * vix_ok
    return s * (d.C[:, -1] / d.O[:, 0] - 1) - np.abs(s) * 2 * cost


def news_by_session(dates: np.ndarray, news: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean headline score and count per session; weekend/holiday headlines roll to the next session.
    A headline dated after a gap of more than 4 days to the next session we hold is dropped (data hole)."""
    sess = dates.astype("datetime64[D]")
    nd = news.index.values.astype("datetime64[D]")
    pos = np.searchsorted(sess, nd, side="left")
    ok = pos < len(sess)
    ok[ok] &= (sess[pos[ok]] - nd[ok]).astype(int) <= 4
    tot, cnt = np.zeros(len(sess)), np.zeros(len(sess))
    np.add.at(tot, pos[ok], (news["mean"] * news["count"]).values[ok])
    np.add.at(cnt, pos[ok], news["count"].values[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan), cnt


def strat_news(d: Day, cost: float, vix_ok: np.ndarray, news: pd.DataFrame, *, min_abs: float = 0.5,
               min_count: int = 2, fade: bool = False) -> np.ndarray:
    """Overnight/pre-market headlines with strong FinBERT tone: trade open -> close in (or against) their direction."""
    out = np.zeros(len(d.dates))
    if news.empty:
        return out
    m, c = news_by_session(d.dates, news)
    s = np.where((np.abs(np.nan_to_num(m)) >= min_abs) & (c >= min_count), np.sign(np.nan_to_num(m)), 0.0)
    s = (-s if fade else s) * vix_ok
    return s * (d.C[:, -1] / d.O[:, 0] - 1) - np.abs(s) * 2 * cost


# ----------------------------------------------------------------------------------------------- stats

def stats(daily: pd.Series) -> dict:
    """daily = portfolio net return per session (0 on days with no trades)."""
    daily = daily.dropna()
    if len(daily) < 20:
        return {"n_days": int(len(daily))}
    mu, sd = daily.mean(), daily.std()
    eq = (1 + daily).cumprod()
    years = daily.groupby(daily.index.year).apply(lambda x: (1 + x).prod() - 1)
    active = daily[daily != 0]
    return {
        "n_days": int(len(daily)), "active_days": int(len(active)),
        "ann_return": float((1 + daily).prod() ** (252 / len(daily)) - 1),
        "sharpe": float(mu / sd * math.sqrt(252)) if sd > 0 else 0.0,
        "t_stat": float(mu / sd * math.sqrt(len(daily))) if sd > 0 else 0.0,
        "hit_rate": float((active > 0).mean()) if len(active) else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "years_positive": f"{int((years > 0).sum())}/{len(years)}",
        "by_year": {int(y): round(float(v), 4) for y, v in years.items()},
    }


@dataclass
class Variant:
    family: str
    name: str
    fn: Callable[..., np.ndarray]
    kw: dict
    vix_min: float = 0.0
    needs_news: bool = False


def variants() -> list[Variant]:
    v: list[Variant] = []
    for vm in (0.0, 20.0):
        tag = f" VIX>={vm:g}" if vm else ""
        v.append(Variant("intraday_momentum", "first 30m -> last 30m" + tag, strat_intraday_momentum, {}, vm))
        v.append(Variant("intraday_momentum", "first 30m -> last 60m" + tag, strat_intraday_momentum,
                         {"entry_slot": 66}, vm))
        for mins in (5, 15, 30):
            for tr in (0.0, 10.0):
                v.append(Variant("opening_range_breakout", f"ORB {mins}m" + (f" target {tr:g}R" if tr else "") + tag,
                                 strat_orb, {"minutes": mins, "target_r": tr}, vm))
            for rv in (1.5, 2.0):
                v.append(Variant("opening_range_breakout", f"ORB {mins}m in-play (vol>={rv:g}x)" + tag,
                                 strat_orb, {"minutes": mins, "rvol_min": rv}, vm))
        v.append(Variant("vwap_trend", "price vs VWAP, long/short" + tag, strat_vwap, {}, vm))
        v.append(Variant("vwap_trend", "price vs VWAP, long only" + tag, strat_vwap, {"long_only": True}, vm))
        for f, s in ((9, 21), (20, 50)):
            v.append(Variant("ema_crossover", f"EMA {f}/{s} 5-min" + tag, strat_ema_cross, {"fast": f, "slow": s}, vm))
        for n, lo, hi in ((14, 30, 70), (2, 10, 90)):
            v.append(Variant("rsi_mean_reversion", f"RSI({n}) {lo:g}/{hi:g}" + tag, strat_rsi_revert,
                             {"n": n, "lo": lo, "hi": hi}, vm))
        for g in (0.005, 0.01):
            for fade in (True, False):
                v.append(Variant("opening_gap", f"gap>={g:.1%} {'fade' if fade else 'follow'}" + tag, strat_gap,
                                 {"min_gap": g, "fade": fade}, vm))
        for th in (0.3, 0.6):
            for fade in (False, True):
                v.append(Variant("news_sentiment", f"pre-market FinBERT |score|>={th} {'fade' if fade else 'follow'}"
                                 + tag, strat_news, {"min_abs": th, "fade": fade}, vm, needs_news=True))
    return v


def evaluate(days: dict[str, Day], vix: pd.Series, news: dict[str, pd.DataFrame], v: Variant,
             cost_mult: float = 1.0, symbols: list[str] | None = None) -> pd.Series:
    """Equal capital per symbol; portfolio return per session = mean of the symbols' net returns that day."""
    cols = {}
    for sym, d in days.items():
        if symbols is not None and sym not in symbols:
            continue
        if v.needs_news and sym not in news:
            continue
        idx = pd.DatetimeIndex(d.dates.astype("datetime64[ns]"))
        vv = vix.reindex(idx, method="ffill").values if len(vix) else np.full(len(idx), np.nan)
        vix_ok = np.ones(len(idx)) if v.vix_min <= 0 else (np.nan_to_num(vv) >= v.vix_min).astype(float)
        kw = dict(v.kw)
        if v.needs_news:
            kw["news"] = news[sym]
        r = v.fn(d, cost_of(sym, cost_mult), vix_ok, **kw)
        cols[sym] = pd.Series(np.nan_to_num(r), index=idx)
    if not cols:
        return pd.Series(dtype=float)
    return pd.DataFrame(cols).mean(axis=1, skipna=True)


def split(s: pd.Series, yrs: range) -> pd.Series:
    return s[s.index.year.isin(list(yrs))]


def main() -> None:
    con = sqlite3.connect(str(INTRADAY_DB))
    days = {s: d for s in UNIVERSE if (d := load(s, con)) is not None}
    vix = load_vix()
    news = {s: n for s in UNIVERSE if not (n := load_news(s)).empty}
    print(f"loaded {len(days)} symbols, news for {sorted(news)}")
    spy = days["SPY"]
    spy_bh = pd.Series(spy.C[:, -1], index=pd.DatetimeIndex(spy.dates.astype("datetime64[ns]"))).pct_change()

    rows = []
    for v in variants():
        for universe, syms in (("all", None), ("etfs", ETFS), ("stocks", [s for s in UNIVERSE if s not in ETFS])):
            s = evaluate(days, vix, news, v, symbols=syms)
            if s.empty:
                continue
            rows.append({"family": v.family, "variant": v.name, "universe": universe,
                         "train": stats(split(s, TRAIN)), "holdout": stats(split(s, HOLDOUT)),
                         "holdout_2x_cost": stats(split(evaluate(days, vix, news, v, 2.0, syms), HOLDOUT)),
                         "holdout_before_costs": stats(split(evaluate(days, vix, news, v, 0.0, syms), HOLDOUT))})
        print(f"  {v.name}")

    # Pick per family on TRAIN Sharpe only, then report that pick's HOLDOUT.
    picks = {}
    for r in rows:
        f = r["family"]
        if r["train"].get("sharpe") is None:
            continue
        if f not in picks or r["train"]["sharpe"] > picks[f]["train"]["sharpe"]:
            picks[f] = r
    n_tests = len(rows)

    def passes(r: dict) -> bool:
        h, h2, t = r["holdout"], r["holdout_2x_cost"], r["train"]
        return (t.get("sharpe", 0) > 0.5 and h.get("sharpe", 0) > 1.0 and h.get("t_stat", 0) > 2.0
                and h2.get("sharpe", 0) > 0.5 and h.get("max_drawdown", -1) > -0.25)

    result = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "train_years": [TRAIN.start, TRAIN.stop - 1], "holdout_years": [HOLDOUT.start, HOLDOUT.stop - 1],
        "symbols": sorted(days), "cost_bps_per_side": {**COST_BPS, "stocks": STOCK_COST_BPS},
        "n_variants_tested": n_tests,
        "pass_rule": "train Sharpe>0.5 AND holdout Sharpe>1 AND holdout t>2 AND holdout Sharpe>0.5 at 2x costs "
                     "AND holdout max drawdown better than -25%",
        "spy_buy_hold": {"train": stats(split(spy_bh, TRAIN)), "holdout": stats(split(spy_bh, HOLDOUT))},
        "picks": {f: {**r, "passes": passes(r)} for f, r in picks.items()},
        "all": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "daytrade.json").write_text(json.dumps(result, indent=1, default=float))
    write_md(result)
    print((OUT / "daytrade.md").read_text())


def write_md(res: dict) -> None:
    def f(x, pct=True):
        return "-" if x is None else (f"{x:+.1%}" if pct else f"{x:.2f}")
    L = [f"# Day-trading study ({res['generated']})", "",
         f"Train {res['train_years'][0]}-{res['train_years'][1]} (variants chosen here), holdout "
         f"{res['holdout_years'][0]}-{res['holdout_years'][1]} (judged here). {len(res['symbols'])} symbols, "
         f"{res['n_variants_tested']} variant x universe combinations tested. Flat every night; costs per side: "
         f"index ETFs 1bp, other ETFs 2bp, stocks 3bp.", "", f"Pass rule: {res['pass_rule']}.", "",
         "| Family | Chosen on train | Universe | Train Sharpe | Holdout ann. return | Holdout Sharpe | t | "
         "Hit rate | Max DD | Years + | 2x cost Sharpe | Before-cost ann. return | Passes |", "|" + "---|" * 13]
    for fam, r in sorted(res["picks"].items(), key=lambda kv: -kv[1]["holdout"].get("sharpe", -9)):
        h, t, h2 = r["holdout"], r["train"], r["holdout_2x_cost"]
        L.append(f"| {fam} | {r['variant']} | {r['universe']} | {f(t.get('sharpe'), False)} | {f(h.get('ann_return'))} "
                 f"| {f(h.get('sharpe'), False)} | {f(h.get('t_stat'), False)} | {f(h.get('hit_rate'))} | "
                 f"{f(h.get('max_drawdown'))} | {h.get('years_positive', '-')} | {f(h2.get('sharpe'), False)} | "
                 f"{f(r['holdout_before_costs'].get('ann_return'))} | {'YES' if r['passes'] else 'no'} |")
    b = res["spy_buy_hold"]["holdout"]
    L += ["", f"Reference, SPY buy-and-hold (holds overnight, so not a day trade): holdout ann. return "
              f"{f(b.get('ann_return'))}, Sharpe {f(b.get('sharpe'), False)}, max DD {f(b.get('max_drawdown'))}.", "",
          "## Best holdout results among ALL variants (for context; picking these after the fact is data-snooping)", "",
          "| Variant | Universe | Train Sharpe | Holdout Sharpe | Holdout ann. return | Before-cost ann. return "
          "| 2x cost Sharpe |", "|---|---|---|---|---|---|---|"]
    top = sorted(res["all"], key=lambda r: -r["holdout"].get("sharpe", -9))[:15]
    for r in top:
        L.append(f"| {r['variant']} | {r['universe']} | {f(r['train'].get('sharpe'), False)} | "
                 f"{f(r['holdout'].get('sharpe'), False)} | {f(r['holdout'].get('ann_return'))} | "
                 f"{f(r['holdout_before_costs'].get('ann_return'))} | {f(r['holdout_2x_cost'].get('sharpe'), False)} |")
    (OUT / "daytrade.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
