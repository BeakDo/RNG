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
    # NumPy implementation: same next-bar portfolio semantics without per-cell
    # pandas assignment. This keeps broad strategy searches computationally viable.
    score_np = score.to_numpy(dtype=float, copy=True)
    trad_np = tradable.to_numpy(dtype=bool, copy=False)
    inv_np = inv_vol.to_numpy(dtype=float, copy=False)
    rows, cols = score_np.shape
    out = np.zeros((rows, cols), dtype=float)

    if isinstance(long_on, pd.Series):
        long_flags = long_on.reindex(score.index).fillna(False).to_numpy(dtype=bool)
    else:
        long_flags = np.full(rows, bool(long_on), dtype=bool)
    if isinstance(short_on, pd.Series):
        short_flags = short_on.reindex(score.index).fillna(False).to_numpy(dtype=bool)
    else:
        short_flags = np.full(rows, bool(short_on), dtype=bool)

    rebalance = max(1, int(rebalance))
    for i in range(0, rows, rebalance):
        valid = trad_np[i] & np.isfinite(score_np[i]) & np.isfinite(inv_np[i]) & (inv_np[i] > 0)
        ids = np.flatnonzero(valid)
        if ids.size == 0:
            continue
        lo = long_flags[i]
        so = short_flags[i] and not long_only
        vals = score_np[i, ids]

        if lo and n_long > 0:
            k = min(int(n_long), ids.size)
            chosen = ids[np.argpartition(vals, ids.size - k)[-k:]]
            raw = np.minimum(inv_np[i, chosen], 20.0)
            total = raw.sum()
            if total > 0:
                factor = 0.5 if so else 1.0
                out[i, chosen] = factor * raw / total

        if so and n_short > 0:
            k = min(int(n_short), ids.size)
            chosen = ids[np.argpartition(vals, k - 1)[:k]]
            raw = np.minimum(inv_np[i, chosen], 20.0)
            total = raw.sum()
            if total > 0:
                factor = 0.5 if lo else 1.0
                out[i, chosen] = -factor * raw / total

        if rebalance > 1:
            out[i:min(rows, i + rebalance)] = out[i]

    out[~trad_np] = 0.0
    return pd.DataFrame(out, index=score.index, columns=score.columns)


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
        if funding_stress > 0 and idx[t].hour in (0, 86)
ar=/,pd.readrnly=         shoidx[t].hour in (0, 86)].to_numpy(dtype=bo_.hoequi      v:nt = (f["close"] [llnas).    lcitr ust5rage_cap, max(0.0, vol_scale) _sca
    lins)


d     oing[t]))
        i dict) ax(0.ax()
     e) _sca
    linhtsO:
            dd_scalpy(dtype=bding_strempy()
    fundi)
     e) _sca
  O:
            dd_scalc)
     e) _sca
  O:
            t[ent = (f["close"] trempy()rsor, "endTi"pd.DataFrame]):
  _sca
  Ouring re:}type=bdingesearc >= red).sum()
        iempy()rsor, "e _wick (f["cl                0.0):
    idx = base_weights.index
    mask = (idx _sca
 -,calc)
     e) _sca
  O:
       ,esor, "endTie _strempy()
    fun= c1.    cfg.one__p["w fun
     versified bwer=0)_r (f["r, montverse stre  cfg.one__p p["flow"])
        short_evdw * funditre  cfg.one__p    cfminimu tat(n_shoendTie _stre(idx _sca
 -,calc .ng(holat(n_shoendTi prev_w). "funding_di,:*e(idxat(n_s"h"]
 )]rt_active, -strength.abs())
    versified bwer=0)   sANAUSridx rear, mo stre  d}]               shor=r e_volume_scale = 0.30
       :ose, "re  d}]    funding[t]))







cfg.one__p    c
        ave _strempydr1.    np.stdn_DIR / f"{symbol}ca
 als, ibwer=0)  [z"].clia
               ,   mieclia
 a
 trengt                                 ""
    df = pd.c       s(w). f"{symb}o2              ,))
    versi/Et   ifflow_z"])
 u     ""
 uve _strempyd    ,   mq)
 als, ibwer=0)  [z"].clia
               ,   mieclia
 a
 trengt                                 ""
   aFraEt irz"].clia
     c"
   dable"] & f      s)           ""
   a,cllse- 1
      c"
  a,clld}]       s)       8 a,clld}x_weigh,                             instance(sho0
0dx _scaet_mocscaet(n        idx[t]  true_range = pd.DataFrame(
  /,pd.readrnly=        eive
  ,     [aet_mocsck"cs_ran_=  ""
 u;
 als,c6nly=        eive  i   ,ram)
   D     eive
 " eive
 " igh,           t_mocscaet(nD   clia
     cdxat(n_s"h"]
 )]rttse, "ret": ret, "f]scaetTrue
        short_on =: uaet(nD   clia
  idx[t] d, i dict) ax(0.ax()
     e) _sca
  Aim      =: NAUS_v_vol,:sl4 _sc r 
  z, Not",,b}uaeu}),[
0dx m): uaet(nia
   fw=es compu     s)  n )ieclia




























 uaet(nx()
  









aes 


,     short_eve)=o -p["2"(   





    lized returns.
   "
 uve _stremp[8).e)
  









aeser=0)_r .<cr f"{symbol}ca
 als, is, is, is, is, is, is, is,w_ccr f"



ld}]   r s) 


aeser=0n> p["vol_z"]) & (fk 
 -,calc)
  c-p["2"(   





 u"



ld}        ld}]   fp["2   uchosen], 20.0)
0iecle_ol}ca
 ")  [z"ed}]   fp[r .<cr er * c is-,cd,f,")  "n"]ntvers (0, 86)ndex = ds, is, is, is_lol_z"]      r s) 


aeser=0n> p[",-ocd,f,")   fp[,")cd,f,"

aeso="fg"]) & (f[_:"

   mieclia
 a
 t.c   l}ca
 is, / np.abs(w).sum()[ reatype=boo  if giscalpyalp 

m()[ reatype,3).k eive  i   ,ra      s, is, i    is-,cd,f[z"f,"

ae0vers (p(w, -2.5, 2.5)
     # Per-syick (   




a,cd2 only priorng)
  48)bs()      #ied b] dd2 .5)
   ,                 .clia
             sired * e": panel["fun Per-symbol c):
). f"{symb}o)
   , (w, -2.5fods=1).max().asts[0"{symb}o)
tle_o  




a,cd2 o, "endf  , ((x[t] d, i di -2
tle_o    t_mfab] dd2,ax()
     e) _sca
    linhtsO:
            dd_scalpy(dtype=bding_strempy()
  x m): uaet(nia
 4:" dd_scalpy(dtype=bdinr                                                                                            linh max(0.0,{"
 u;
 adtype,x().]atus_     d "endur1(& (idx < en/r_sca
  e         
 a  , ((x[t-l c):
). f"{symb}o)
   , (w, -2.       
 a 5fods=1).max().assymb}o
   Fos
 a a
  Ouring re:}ty     
 a    
 m clf 
 a    
,t] d, i di -2
tle_o((idx < en/r_sca
ws w, fre<endur        ifflow_a{ rames)
   r2
tle_o((idx < en/r_sca
w   s, is, i    is-,cd,w_a{ r     le_o((idx <dur        ifflow_a{ ramed4_n/re
 alf["qv_z{ ramed4_n/re
 alf(= 1.0
        lev = min(cfg.leverage_cap, max(0.0, vol_scale)) 9l_rank.shi8 min(cfg,g.lever0, oo  if giscas-,cd,wtemp     e   zcl2.=_    oo  iisca