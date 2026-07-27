#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BARS_PER_YEAR = 365.25 * 6.0  # 4-hour bars
START = pd.Timestamp("2020-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
END_EXCLUSIVE = pd.Timestamp("2026-07-01", tz="UTC")

# Includes later delistings and severe failures to reduce survivor bias.
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","SOLUSDT","DOGEUSDT",
    "DOTUSDT","LINKUSDT","LTCUSDT","BCHUSDT","TRXUSDT","ETCUSDT","EOSUSDT",
    "XLMUSDT","ATOMUSDT","AAVEUSDT","UNIUSDT","SUSHIUSDT","CRVUSDT","FILUSDT",
    "NEARUSDT","AVAXUSDT","RUNEUSDT","FTMUSDT","MATICUSDT","ALGOUSDT","VETUSDT",
    "SANDUSDT","MANAUSDT","AXSUSDT","GALAUSDT","APEUSDT","OPUSDT","ARBUSDT",
    "APTUSDT","INJUSDT","LDOUSDT","DYDXUSDT","FTTUSDT","SRMUSDT","WAVESUSDT",
    "LUNAUSDT","LUNA2USDT","HNTUSDT","RENUSDT","REEFUSDT","YFIUSDT","1INCHUSDT",
    "GRTUSDT","EGLDUSDT","KSMUSDT","ZECUSDT","DASHUSDT","XMRUSDT",
]

DATA_DIR = Path("research/data")
OUT_DIR = Path("research/results")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "alpha-research/1.0"})


def month_iter(start: pd.Timestamp, end_exclusive: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    while cur < end_exclusive:
        yield cur.year, cur.month
        cur += pd.offsets.MonthBegin(1)


def get_bytes(url: str, timeout: int = 30) -> bytes | None:
    for attempt in range(5):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 4:
                return None
            time.sleep(0.4 * (2 ** attempt))
    return None


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    key = f"{year:04d}-{month:02d}"
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/4h/{symbol}-4h-{key}.zip"
    )
    raw = get_bytes(url)
    if not raw:
        return None
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                df = pd.read_csv(fh, header=None)
    except Exception:
        return None
    if df.empty or df.shape[1] < 11:
        return None
    df = df.iloc[:, :12]
    df.columns = [
        "open_time","open","high","low","close","volume","close_time",
        "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore",
    ]
    for c in ["open","high","low","close","volume","quote_volume","trades","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Binance archives changed timestamp precision for some newer files.
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    unit = "us" if ts.dropna().median() > 1e14 else "ms"
    df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return df[["timestamp","open","high","low","close","quote_volume","trades","taker_buy_quote"]].dropna()


def download_symbol(symbol: str) -> str:
    target = DATA_DIR / f"{symbol}_4h.parquet"
    if target.exists():
        return symbol
    frames = []
    for year, month in month_iter(START, END_EXCLUSIVE):
        frame = download_month(symbol, year, month)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return ""
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[(df["timestamp"] >= START) & (df["timestamp"] < END_EXCLUSIVE)]
    if len(df) < 500:
        return ""
    df.to_parquet(target, index=False, compression="zstd")
    return symbol


def download_all() -> list[str]:
    existing = [s for s in SYMBOLS if (DATA_DIR / f"{s}_4h.parquet").exists()]
    missing = [s for s in SYMBOLS if s not in existing]
    good = set(existing)
    if missing:
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(download_symbol, s): s for s in missing}
            for fut in as_completed(futs):
                s = fut.result()
                if s:
                    good.add(s)
                    print("downloaded", s, flush=True)
    return sorted(good)


def fetch_funding_symbol(symbol: str) -> pd.DataFrame:
    target = DATA_DIR / f"{symbol}_funding.parquet"
    if target.exists():
        return pd.read_parquet(target)
    rows = []
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END_EXCLUSIVE.timestamp() * 1000) - 1
    cursor = start_ms
    while cursor < end_ms:
        try:
            r = SESSION.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
                timeout=20,
            )
            if r.status_code in (400, 404, 451):
                break
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.08)
        if len(data) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp","funding_rate"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["timestamp","funding_rate"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    df.to_parquet(target, index=False, compression="zstd")
    return df


def fetch_all_funding(symbols: list[str]):
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(fetch_funding_symbol, symbols))


@dataclass(frozen=True)
class Config:
    family: str
    params: dict[str, Any]
    target_vol: float = 0.60
    leverage_cap: float = 4.0
    one_way_cost: float = 0.0008
    adverse_funding: float = 0.0


def load_panel(symbols: list[str]):
    frames = {}
    all_index = None
    for s in symbols:
        p = DATA_DIR / f"{s}_4h.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.set_index("timestamp").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        frames[s] = d
        all_index = d.index if all_index is None else all_index.union(d.index)
    idx = pd.date_range(START, END_EXCLUSIVE - pd.Timedelta(hours=4), freq="4h", tz="UTC")
    symbols = sorted(frames)
    cols = ["open","high","low","close","quote_volume","trades","taker_buy_quote"]
    panel = {c: pd.DataFrame(index=idx, columns=symbols, dtype=float) for c in cols}
    for s, d in frames.items():
        for c in cols:
            panel[c].loc[d.index.intersection(idx), s] = d.loc[d.index.intersection(idx), c]
    funding = pd.DataFrame(0.0, index=idx, columns=symbols)
    funding_available = pd.DataFrame(False, index=idx, columns=symbols)
    for s in symbols:
        p = DATA_DIR / f"{s}_funding.parquet"
        if not p.exists():
            continue
        fd = pd.read_parquet(p)
        if fd.empty:
            continue
        fd["timestamp"] = pd.to_datetime(fd["timestamp"], utc=True)
        ser = fd.set_index("timestamp")["funding_rate"].sort_index()
        # Funding timestamps align to 00/08/16 UTC; map to nearest 4h bar.
        ser.index = ser.index.floor("4h")
        ser = ser[~ser.index.duplicated(keep="last")]
        common = ser.index.intersection(idx)
        funding.loc[common, s] = ser.loc[common]
        funding_available.loc[common, s] = True
    panel["funding"] = funding
    panel["funding_available"] = funding_available
    return symbols, panel


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True, method="average")


def rolling_z(x: pd.DataFrame, n: int) -> pd.DataFrame:
    m = x.rolling(n, min_periods=max(12, n // 3)).mean()
    sd = x.rolling(n, min_periods=max(12, n // 3)).std().replace(0, np.nan)
    return (x - m) / sd


def build_features(panel: dict[str, pd.DataFrame]):
    close = panel["close"]
    high, low, open_ = panel["high"], panel["low"], panel["open"]
    qv = panel["quote_volume"]
    tbq = panel["taker_buy_quote"]
    ret = close.pct_change()
    logret = np.log(close).diff()
    flow = (2.0 * tbq / qv.replace(0, np.nan) - 1.0).clip(-1, 1)
    flow3 = flow.rolling(3, min_periods=1).mean()
    vol18 = logret.rolling(18, min_periods=12).std() * math.sqrt(BARS_PER_YEAR)
    vol42 = logret.rolling(42, min_periods=24).std() * math.sqrt(BARS_PER_YEAR)
    vol126 = logret.rolling(126, min_periods=42).std() * math.sqrt(BARS_PER_YEAR)
    qv_med = qv.rolling(42, min_periods=18).median()
    liq_rank = cs_rank(qv_med)
    qv_z = rolling_z(np.log1p(qv), 126)
    flow_z = rolling_z(flow3, 126)
    r6 = close.pct_change(6)
    r18 = close.pct_change(18)
    r42 = close.pct_change(42)
    r126 = close.pct_change(126)
    market_ret = ret.median(axis=1)
    market_mom = (1 + market_ret.fillna(0)).rolling(42).apply(np.prod, raw=True) - 1
    breadth = (r42 > 0).mean(axis=1)
    btc = close["BTCUSDT"] if "BTCUSDT" in close else close.median(axis=1)
    btc_fast = btc.ewm(span=18, adjust=False).mean()
    btc_slow = btc.ewm(span=72, adjust=False).mean()
    bull = (btc_fast > btc_slow) & (breadth > 0.52)
    bear = (btc_fast < btc_slow) & (breadth < 0.48)
    body_hi = pd.DataFrame(np.maximum(open_.values, close.values), index=close.index, columns=close.columns)
    body_lo = pd.DataFrame(np.minimum(open_.values, close.values), index=close.index, columns=close.columns)
    upper_wick = (high - body_hi) / close.replace(0, np.nan)
    lower_wick = (body_lo - low) / close.replace(0, np.nan)
    true_range = pd.DataFrame(
        np.maximum.reduce([
            (high-low).values,
            (high-close.shift(1)).abs().values,
            (low-close.shift(1)).abs().values,
        ]), index=close.index, columns=close.columns)
    atr = true_range.rolling(18, min_periods=12).mean() / close
    tradable = (
        close.notna()
        & close.shift(126).notna()
        & (liq_rank >= 0.30)
        & (qv_med >= 1_000_000)
    )
    return {
        "close": close, "ret": ret, "flow": flow3, "flow_z": flow_z,
        "vol18": vol18, "vol42": vol42, "vol126": vol126,
        "qv_z": qv_z, "liq_rank": liq_rank, "r6": r6, "r18": r18,
        "r42": r42, "r126": r126, "bull": bull, "bear": bear,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        "atr": atr, "tradable": tradable, "funding": panel["funding"],
        "funding_available": panel["funding_available"],
    }


def select_weights(score: pd.DataFrame, tradable: pd.DataFrame, inv_vol: pd.DataFrame,
                   n_long: int, n_short: int, long_on: pd.Series | bool = True,
                   short_on: pd.Series | bool = True, rebalance: int = 1,
                   long_only: bool = False) -> pd.DataFrame:
    score = score.where(tradable)
    weights = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for i in range(0, len(score), rebalance):
        # An explicit zero row at every rebalance allows the strategy to exit.
        weights.iloc[i] = 0.0
        row = score.iloc[i].dropna()
        if row.empty:
            continue
        lo = bool(long_on.iloc[i]) if isinstance(long_on, pd.Series) else bool(long_on)
        so = bool(short_on.iloc[i]) if isinstance(short_on, pd.Series) else bool(short_on)
        if lo and n_long > 0:
            names = row.nlargest(min(n_long, len(row))).index
            raw = inv_vol.iloc[i][names].replace([np.inf, -np.inf], np.nan).dropna().clip(upper=20)
            if len(raw):
                weights.loc[score.index[i], raw.index] = 0.5 * raw / raw.sum() if (so and not long_only) else raw / raw.sum()
        if so and not long_only and n_short > 0:
            names = row.nsmallest(min(n_short, len(row))).index
            raw = inv_vol.iloc[i][names].replace([np.inf, -np.inf], np.nan).dropna().clip(upper=20)
            if len(raw):
                weights.loc[score.index[i], raw.index] = -0.5 * raw / raw.sum() if lo else -raw / raw.sum()
    # Hold weights until next rebalance, but zero symbols that become unavailable.
    weights = weights.ffill(limit=max(0, rebalance-1)).fillna(0.0)
    return weights.where(tradable, 0.0)


def event_weights(long_event: pd.DataFrame, short_event: pd.DataFrame, strength: pd.DataFrame,
                  f: dict, hold: int, n_long: int, n_short: int,
                  long_regime: pd.Series | bool = True, short_regime: pd.Series | bool = True):
    long_active = long_event.rolling(hold, min_periods=1).max().astype(bool)
    short_active = short_event.rolling(hold, min_periods=1).max().astype(bool)
    score = strength.where(long_active, np.nan).where(~short_active, -strength.abs())
    # Ensure short events receive negative scores even when strength itself is positive.
    score = score.mask(short_active, -strength.abs())
    return select_weights(score, f["tradable"], 1/f["vol42"].replace(0, np.nan),
                          n_long, n_short, long_regime, short_regime, 1)


def make_base_weights(cfg: Config, f: dict) -> pd.DataFrame:
    p = cfg.params
    inv_vol = 1 / f["vol42"].replace(0, np.nan)
    fam = cfg.family
    if fam == "cross_momentum":
        h = p["h"]
        mom = f[f"r{h}"]
        short_mom = f["r6"] if p.get("skip", 0) == 0 else f[f"r{h}"] - f["r6"]
        score = cs_rank(mom) + p["flow_w"] * cs_rank(f["flow_z"]) + 0.35 * cs_rank(f["qv_z"])
        score = score - 0.25 * cs_rank(f["vol18"])
        long_on = f["bull"] if p["regime"] else True
        short_on = f["bear"] if p["regime"] else True
        return select_weights(score, f["tradable"], inv_vol, p["n"], p["n"], long_on, short_on, p["reb"])

    if fam == "residual_momentum":
        h = p["h"]
        mom = f[f"r{h}"]
        mkt = mom.median(axis=1)
        residual = mom.sub(mkt, axis=0)
        score = cs_rank(residual) + p["flow_w"] * cs_rank(f["flow_z"])
        return select_weights(score, f["tradable"], inv_vol, p["n"], p["n"], f["bull"], f["bear"], p["reb"])

    if fam == "compression_breakout":
        w = p["breakout"]
        prior_hi = f["close"].shift(1).rolling(w).max()
        prior_lo = f["close"].shift(1).rolling(w).min()
        vol_rank = f["vol18"].rolling(126, min_periods=42).rank(pct=True)
        comp = vol_rank.shift(1).rolling(6, min_periods=1).min() < p["comp_q"]
        long_event = (f["close"] > prior_hi) & comp & (f["qv_z"] > p["vol_z"]) & (f["flow"] > p["flow"])
        short_event = (f["close"] < prior_lo) & comp & (f["qv_z"] > p["vol_z"]) & (f["flow"] < -p["flow"])
        strength = f["qv_z"].clip(-3,3) + f["flow_z"].clip(-3,3).abs()
        return event_weights(long_event, short_event, strength, f, p["hold"], p["n"], p["n"], f["bull"], f["bear"])

    if fam == "flow_continuation":
        rz = rolling_z(f["ret"], 126)
        long_event = (rz > p["rz"]) & (f["qv_z"] > p["vol_z"]) & (f["flow"] > p["flow"])
        short_event = (rz < -p["rz"]) & (f["qv_z"] > p["vol_z"]) & (f["flow"] < -p["flow"])
        strength = rz.abs() + f["qv_z"].clip(lower=0) + f["flow_z"].abs()
        return event_weights(long_event, short_event, strength, f, p["hold"], p["n"], p["n"], f["bull"], f["bear"])

    if fam == "exhaustion_reversal":
        rz = rolling_z(f["ret"], 126)
        long_event = (rz < -p["rz"]) & (f["qv_z"] > p["vol_z"]) & (f["lower_wick"] > p["wick"])
        short_event = (rz > p["rz"]) & (f["qv_z"] > p["vol_z"]) & (f["upper_wick"] > p["wick"])
        strength = rz.abs() + f["qv_z"].clip(lower=0) + 10*(f["lower_wick"] + f["upper_wick"])
        # Reversal is allowed in either market regime, but only small diversified baskets.
        return event_weights(long_event, short_event, strength, f, p["hold"], p["n"], p["n"], True, True)

    if fam == "funding_dislocation":
        fr = f["funding"].rolling(3, min_periods=1).mean()
        trend = f[f"r{p['h']}"]
        # Positive score: negative funding plus positive trend. Negative score: positive funding plus negative trend.
        score = -cs_rank(fr) + cs_rank(trend) + p["flow_w"] * cs_rank(f["flow_z"])
        available = f["tradable"] & f["funding_available"].rolling(3, min_periods=1).max().astype(bool)
        return select_weights(score, available, inv_vol, p["n"], p["n"], True, True, p["reb"])

    raise ValueError(f"unknown family {fam}")


def backtest(base_weights: pd.DataFrame, f: dict, cfg: Config,
             start: pd.Timestamp, end: pd.Timestamp, funding_stress: float = 0.0):
    idx = base_weights.index
    mask = (idx >= start) & (idx < end)
    locs = np.flatnonzero(mask)
    if len(locs) < 100:
        return None, None
    first, last = locs[0], locs[-1]
    returns = f["ret"].fillna(0.0).to_numpy()
    base = base_weights.fillna(0.0).to_numpy()
    funding = f["funding"].fillna(0.0).to_numpy()
    funding_av = f["funding_available"].to_numpy(dtype=bool)

    equity = 1.0
    peak = 1.0
    prev_w = np.zeros(base.shape[1])
    realized = []
    records = []
    for t in range(first, last + 1):
        # Signal from t-1 controls position during return t.
        desired = base[t-1].copy() if t > 0 else np.zeros(base.shape[1])
        gross = np.abs(desired).sum()
        if gross > 0:
            desired /= max(1.0, gross)  # base portfolio gross <= 1

        # Volatility target uses only prior realized returns.
        if len(realized) >= 42:
            rv = np.std(realized[-42:], ddof=1) * math.sqrt(BARS_PER_YEAR)
            vol_scale = cfg.target_vol / max(rv, 0.08)
        else:
            vol_scale = 0.5
        dd = equity / peak - 1.0
        if dd <= -0.25:
            dd_scale = 0.08
        elif dd <= -0.18:
            dd_scale = 0.30
        elif dd <= -0.10:
            dd_scale = 0.65
        else:
            dd_scale = 1.0
        lev = min(cfg.leverage_cap, max(0.0, vol_scale)) * dd_scale
        w = desired * lev
        # Per-symbol cap guards against a single illiquid name dominating.
        w = np.clip(w, -2.5, 2.5)
        if np.abs(w).sum() > cfg.leverage_cap:
            w *= cfg.leverage_cap / np.abs(w).sum()

        gross_ret = float(np.nansum(w * returns[t]))
        turnover = float(np.abs(w - prev_w).sum())
        cost = turnover * cfg.one_way_cost
        # Funding entries exist only on settlement bars. Missing observations receive
        # an adverse stress only in the explicit stress test.
        fund = -float(np.nansum(w * funding[t]))
        if funding_stress > 0 and idx[t].hour in (0, 8, 16):
            missing_exposure = float(np.abs(w[~funding_av[t]]).sum())
            fund -= missing_exposure * funding_stress
        bar_ret = max(-0.95, gross_ret - cost + fund)
        equity *= 1.0 + bar_ret
        peak = max(peak, equity)
        realized.append(bar_ret)
        records.append((idx[t], equity, bar_ret, equity/peak-1.0, turnover, np.abs(w).sum()))
        prev_w = w
    curve = pd.DataFrame(records, columns=["timestamp","equity","return","drawdown","turnover","gross_leverage"]).set_index("timestamp")
    years = len(curve) / BARS_PER_YEAR
    cagr = max(curve.equity.iloc[-1], 1e-12) ** (1/max(years, 1e-9)) - 1
    mdd = abs(float(curve.drawdown.min()))
    ann_vol = curve["return"].std() * math.sqrt(BARS_PER_YEAR)
    sharpe = curve["return"].mean() * BARS_PER_YEAR / ann_vol if ann_vol > 0 else -99
    stats = {
        "cagr": float(cagr), "mdd": mdd, "sharpe": float(sharpe),
        "total_return": float(curve.equity.iloc[-1]-1),
        "annual_turnover": float(curve.turnover.mean()*BARS_PER_YEAR),
        "max_gross_leverage": float(curve.gross_leverage.max()),
        "years": float(years),
    }
    return curve, stats


def configs() -> list[Config]:
    out = []
    for h, n, reb, flow_w, regime in itertools.product([18,42,126],[2,4],[1,3],[0.0,0.6],[True,False]):
        out.append(Config("cross_momentum", {"h":h,"n":n,"reb":reb,"flow_w":flow_w,"regime":regime,"skip":6}))
    for h, n, reb, flow_w in itertools.product([18,42,126],[2,4],[1,3],[0.0,0.5]):
        out.append(Config("residual_momentum", {"h":h,"n":n,"reb":reb,"flow_w":flow_w}))
    for br, cq, vz, fl, hold, n in itertools.product([18,42,84],[0.2,0.35],[0.5,1.0],[0.10],[6,12,24],[1,2]):
        out.append(Config("compression_breakout", {"breakout":br,"comp_q":cq,"vol_z":vz,"flow":fl,"hold":hold,"n":n}))
    for rz, vz, fl, hold, n in itertools.product([1.5,2.0,2.5],[0.5,1.0],[0.12],[3,6,12],[1,2]):
        out.append(Config("flow_continuation", {"rz":rz,"vol_z":vz,"flow":fl,"hold":hold,"n":n}))
    for rz, vz, wick, hold, n in itertools.product([1.8,2.4,3.0],[0.5,1.0],[0.008,0.015],[2,3,6],[1]):
        out.append(Config("exhaustion_reversal", {"rz":rz,"vol_z":vz,"wick":wick,"hold":hold,"n":n}))
    for h,n,reb,fw in itertools.product([18,42],[2,4],[1,3],[0.0,0.4]):
        out.append(Config("funding_dislocation", {"h":h,"n":n,"reb":reb,"flow_w":fw}))
    return out


def folds():
    return [
        (pd.Timestamp("2021-01-01",tz="UTC"), pd.Timestamp("2022-01-01",tz="UTC")),
        (pd.Timestamp("2022-01-01",tz="UTC"), pd.Timestamp("2023-01-01",tz="UTC")),
        (pd.Timestamp("2023-01-01",tz="UTC"), pd.Timestamp("2024-01-01",tz="UTC")),
        (pd.Timestamp("2024-01-01",tz="UTC"), HOLDOUT_START),
    ]


def score_config(cfg: Config, f: dict):
    base = make_base_weights(cfg, f)
    rows = []
    for a,b in folds():
        _, st = backtest(base, f, cfg, a, b)
        if st is None:
            return None
        rows.append(st)
    c = np.array([x["cagr"] for x in rows])
    d = np.array([x["mdd"] for x in rows])
    sh = np.array([x["sharpe"] for x in rows])
    # Strongly favor repeatable fold performance, not a single explosive year.
    score = (
        2.0*np.median(np.log1p(np.clip(c,-0.95,30)))
        +0.8*np.quantile(np.log1p(np.clip(c,-0.95,30)),0.25)
        +0.25*np.median(sh)
        -12*max(0,float(d.max())-0.25)
        -0.45*np.std(np.clip(c,-0.95,30))
        -0.0005*np.median([x["annual_turnover"] for x in rows])
    )
    return {"score":float(score),"median_cagr":float(np.median(c)),"worst_cagr":float(c.min()),
            "worst_mdd":float(d.max()),"median_sharpe":float(np.median(sh)),"folds":rows,
            "config":asdict(cfg)}, base


def leverage_refine(candidate: dict, f: dict):
    base_cfg = Config(**candidate["config"])
    base = make_base_weights(base_cfg, f)
    options=[]
    for tv,cap in itertools.product([0.45,0.60,0.80,1.00,1.25,1.50,2.00],[2.0,3.0,4.0,6.0,8.0]):
        cfg=Config(base_cfg.family,base_cfg.params,tv,cap,base_cfg.one_way_cost,base_cfg.adverse_funding)
        sts=[]
        for a,b in folds():
            _,st=backtest(base,f,cfg,a,b,funding_stress=0.0001)
            sts.append(st)
        c=np.array([x["cagr"] for x in sts]); d=np.array([x["mdd"] for x in sts])
        feasible=d.max()<=0.27
        score=(np.median(np.log1p(np.clip(c,-.95,50)))+0.5*np.quantile(np.log1p(np.clip(c,-.95,50)),.25)
               -20*max(0,float(d.max())-.27)-0.5*np.std(np.clip(c,-.95,50)))
        options.append({"score":float(score),"feasible":bool(feasible),"median_cagr":float(np.median(c)),
                        "worst_cagr":float(c.min()),"worst_mdd":float(d.max()),"folds":sts,"config":asdict(cfg)})
    feasible=[x for x in options if x["feasible"]]
    chosen=max(feasible or options,key=lambda x:x["score"])
    return chosen, sorted(options,key=lambda x:x["score"],reverse=True)


def parameter_neighbors(cfg: Config):
    neighbors=[]
    p=cfg.params.copy()
    numeric=[k for k,v in p.items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
    for k in numeric:
        for factor in (0.85,1.15):
            q=p.copy(); val=q[k]
            q[k]=max(1,round(val*factor)) if isinstance(val,int) else val*factor
            neighbors.append(Config(cfg.family,q,cfg.target_vol,cfg.leverage_cap,cfg.one_way_cost,cfg.adverse_funding))
    return neighbors[:20]


def run_search():
    symbols=download_all()
    fetch_all_funding(symbols)
    symbols,panel=load_panel(symbols)
    f=build_features(panel)
    rows=[]; bases={}
    all_cfg=configs()
    print("symbols",len(symbols),"configs",len(all_cfg),flush=True)
    for i,cfg in enumerate(all_cfg,1):
        try:
            res,base=score_config(cfg,f)
            if res:
                key=json.dumps(res["config"],sort_keys=True)
                rows.append(res); bases[key]=base
        except Exception as e:
            print("config error",cfg.family,e,flush=True)
        if i%100==0:
            print("tested",i,flush=True)
    rows.sort(key=lambda x:x["score"],reverse=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ("folds","config")} | {"family":r["config"]["family"],"params":json.dumps(r["config"]["params"],sort_keys=True)} for r in rows]).to_csv(OUT_DIR/"leaderboard.csv",index=False)

    # Family champions and fixed equal-weight ensembles. All selection remains pre-holdout.
    champions=[]
    for fam in sorted({r["config"]["family"] for r in rows}):
        champions.append(next(r for r in rows if r["config"]["family"]==fam))
    champions.sort(key=lambda x: x["score"], reverse=True)
    candidates=rows[:15]
    for n in (3,4,5,6):
        chosen=champions[:n] if len(champions)>=n else champions
        if len(chosen)<2: continue
        matrices=[]
        for r in chosen:
            cfg=Config(**r["config"]); matrices.append(make_base_weights(cfg,f))
        ensemble=sum(matrices)/len(matrices)
        cfg=Config("ensemble",{"members":[r["config"] for r in chosen]},0.60,4.0)
        sts=[]
        for a,b in folds():
            _,st=backtest(ensemble,f,cfg,a,b)
            sts.append(st)
        c=np.array([x["cagr"] for x in sts]); d=np.array([x["mdd"] for x in sts]); sh=np.array([x["sharpe"] for x in sts])
        score=2*np.median(np.log1p(np.clip(c,-.95,30)))+0.8*np.quantile(np.log1p(np.clip(c,-.95,30)),.25)+.25*np.median(sh)-12*max(0,float(d.max())-.25)-.45*np.std(np.clip(c,-.95,30))
        candidates.append({"score":float(score),"median_cagr":float(np.median(c)),"worst_cagr":float(c.min()),"worst_mdd":float(d.max()),"median_sharpe":float(np.median(sh)),"folds":sts,"config":asdict(cfg),"_ensemble_base":ensemble})
    candidates.sort(key=lambda x:x["score"],reverse=True)

    robust=[]
    for cand in candidates[:12]:
        if cand["config"]["family"]=="ensemble":
            refined_cfg=Config(**cand["config"]); base=cand["_ensemble_base"]
            # fixed risk grid for ensemble
            opts=[]
            for tv,cap in itertools.product([.45,.6,.8,1.0,1.25,1.5,2.0],[2.,3.,4.,6.,8.]):
                cfg=Config("ensemble",refined_cfg.params,tv,cap)
                sts=[backtest(base,f,cfg,a,b,0.0001)[1] for a,b in folds()]
                c=np.array([x["cagr"] for x in sts]); d=np.array([x["mdd"] for x in sts])
                sc=np.median(np.log1p(np.clip(c,-.95,50)))+.5*np.quantile(np.log1p(np.clip(c,-.95,50)),.25)-20*max(0,float(d.max())-.27)-.5*np.std(np.clip(c,-.95,50))
                opts.append({"score":float(sc),"feasible":bool(d.max()<=.27),"median_cagr":float(np.median(c)),"worst_cagr":float(c.min()),"worst_mdd":float(d.max()),"folds":sts,"config":asdict(cfg)})
            feas=[x for x in opts if x["feasible"]]; ref=max(feas or opts,key=lambda x:x["score"])
            neighbor_rate=1.0
        else:
            ref,_=leverage_refine(cand,f)
            cfg=Config(**ref["config"])
            neigh=parameter_neighbors(cfg)
            good=0
            for ng in neigh:
                try:
                    b=make_base_weights(ng,f)
                    sts=[backtest(b,f,ng,a,z,0.0001)[1] for a,z in folds()]
                    if max(x["mdd"] for x in sts)<=.30 and np.median([x["cagr"] for x in sts])>=max(0,ref["median_cagr"]*.4): good+=1
                except Exception: pass
            neighbor_rate=good/max(1,len(neigh))
        robust_score=ref["score"]+1.2*neighbor_rate-8*max(0,ref["worst_mdd"]-.27)
        robust.append({"robust_score":float(robust_score),"neighbor_pass_rate":float(neighbor_rate),"base_candidate":{k:v for k,v in cand.items() if not k.startswith("_")},"refined":ref})
    robust.sort(key=lambda x:x["robust_score"],reverse=True)
    frozen=robust[0]
    frozen["symbols"]=symbols
    frozen["data_start"]=str(START); frozen["preholdout_end"]=str(HOLDOUT_START); frozen["holdout_start"]=str(HOLDOUT_START); frozen["data_end_exclusive"]=str(END_EXCLUSIVE)
    frozen["holdout_used_in_selection"]=False
    frozen["tested_config_count"]=len(all_cfg)
    frozen["created_at_utc"]=datetime.now(timezone.utc).isoformat()
    (OUT_DIR/"frozen_candidate.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    (OUT_DIR/"robust_candidates.json").write_text(json.dumps(robust,indent=2),encoding="utf-8")
    summary={"symbols":symbols,"tested":len(all_cfg),"family_champions":champions,"frozen":frozen}
    (OUT_DIR/"search_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("FROZEN",json.dumps(frozen["refined"],indent=2),flush=True)


def reconstruct_base(cfg: Config,f:dict):
    if cfg.family!="ensemble": return make_base_weights(cfg,f)
    mats=[]
    for member in cfg.params["members"]:
        mats.append(make_base_weights(Config(**member),f))
    return sum(mats)/len(mats)


def run_holdout():
    lock=OUT_DIR/"HOLDOUT_OPENED.lock"
    if lock.exists():
        raise RuntimeError("holdout already opened")
    frozen=json.loads((OUT_DIR/"frozen_candidate.json").read_text())
    symbols,panel=load_panel(frozen["symbols"])
    f=build_features(panel)
    cfg=Config(**frozen["refined"]["config"])
    base=reconstruct_base(cfg,f)
    curve,st=backtest(base,f,cfg,HOLDOUT_START,END_EXCLUSIVE,0.0)
    _,stress=backtest(base,f,cfg,HOLDOUT_START,END_EXCLUSIVE,0.00015)
    result={"target":{"cagr":10.0,"mdd":0.30},"holdout":st,"cost_funding_stress":stress,
            "pass":bool(st["cagr"]>=10 and st["mdd"]<=.30 and stress["mdd"]<=.30),
            "opened_at_utc":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg)}
    curve.to_csv(OUT_DIR/"holdout_curve.csv")
    (OUT_DIR/"holdout_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lock.write_text(result["opened_at_utc"])
    print("HOLDOUT",json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["search","holdout"])
    args=ap.parse_args()
    if args.mode=="search": run_search()
    else: run_holdout()

if __name__=="__main__": main()
