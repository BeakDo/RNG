#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOLS = [
    "DOGEUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT", "WIFUSDT",
    "1000PEPEUSDT", "TRBUSDT", "LABUSDT", "RIVERUSDT",
]
TIMEFRAMES = ["5m", "1m"]
MONTHS = pd.period_range("2024-01", "2026-06", freq="M").astype(str).tolist()
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALID_END = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-07-01", tz="UTC")
COST = 0.0020
TP = 0.03
SL = 0.03
OUT = Path("results")
CACHE = Path("cache")
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def download_month(symbol: str, tf: str, month: str) -> Path | None:
    d = CACHE / symbol / tf
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{symbol}-{tf}-{month}.zip"
    if path.exists() and path.stat().st_size > 100:
        return path
    url = f"{BASE}/{symbol}/{tf}/{symbol}-{tf}-{month}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if len(r.content) < 100:
                return None
            path.write_bytes(r.content)
            return path
        except Exception as e:
            if attempt == 3:
                print(f"DOWNLOAD_FAIL {url}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def read_month(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            return pd.DataFrame(columns=COLS)
        with zf.open(names[0]) as f:
            raw = pd.read_csv(f, header=None)
    if raw.shape[1] < 12:
        return pd.DataFrame(columns=COLS)
    raw = raw.iloc[:, :12]
    raw.columns = COLS
    raw = raw[pd.to_numeric(raw["open_time"], errors="coerce").notna()].copy()
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce").astype("int64")
    unit = "us" if raw["open_time"].median() > 10**14 else "ms"
    raw["timestamp"] = pd.to_datetime(raw["open_time"], unit=unit, utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]]


def load_symbol(symbol: str, tf: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    found = 0
    for month in MONTHS:
        p = download_month(symbol, tf, month)
        if p is None:
            continue
        try:
            x = read_month(p)
            if not x.empty:
                parts.append(x)
                found += 1
        except Exception as e:
            print(f"READ_FAIL {p}: {e}")
    if not parts:
        print(f"NO_DATA {symbol} {tf}")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC")) & (df["timestamp"] < TEST_END)]
    print(f"LOADED {symbol} {tf}: {len(df):,} bars, {found} months, {df.timestamp.min()} -> {df.timestamp.max()}")
    return df


def first_hit(df: pd.DataFrame, signal_i: int, side: str, max_hold: int) -> dict | None:
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None
    entry = float(df.at[entry_i, "open"])
    if not math.isfinite(entry) or entry <= 0:
        return None
    if side == "long":
        tp_price = entry * (1 + TP)
        sl_price = entry * (1 - SL)
    else:
        tp_price = entry * (1 - TP)
        sl_price = entry * (1 + SL)
    end_i = min(len(df) - 1, entry_i + max_hold - 1)
    exit_i = end_i
    raw_return = None
    reason = "timeout"
    for j in range(entry_i, end_i + 1):
        hi = float(df.at[j, "high"])
        lo = float(df.at[j, "low"])
        if side == "long":
            hit_tp, hit_sl = hi >= tp_price, lo <= sl_price
        else:
            hit_tp, hit_sl = lo <= tp_price, hi >= sl_price
        if hit_sl:
            raw_return, reason, exit_i = -SL, "sl", j
            break
        if hit_tp:
            raw_return, reason, exit_i = TP, "tp", j
            break
    if raw_return is None:
        exit_price = float(df.at[exit_i, "close"])
        raw_return = (exit_price / entry - 1) if side == "long" else (entry / exit_price - 1)
    return {
        "entry_i": entry_i, "exit_i": exit_i,
        "entry_time": df.at[entry_i, "timestamp"], "exit_time": df.at[exit_i, "timestamp"],
        "entry": entry, "raw_return": float(raw_return),
        "net_return": float(raw_return - COST), "reason": reason,
    }


def feature_candidates(df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    minutes = 5 if tf == "5m" else 1
    pump_bars = int(180 / minutes)
    base_bars = int(360 / minutes)
    max_hold = int(120 / minutes)
    candidates: list[dict] = []
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].replace(0, np.nan)
    candle_range = (h - l).replace(0, np.nan)
    body_ratio = (c - o).abs() / candle_range
    close_loc = (c - l) / candle_range
    for box_minutes in [30, 60, 90, 120]:
        box = int(box_minutes / minutes)
        if len(df) < base_bars + pump_bars + box + max_hold + 10:
            continue
        box_high = h.shift(1).rolling(box).max()
        box_low = l.shift(1).rolling(box).min()
        box_mid = c.shift(1).rolling(box).median()
        box_vol_mean = v.shift(1).rolling(box).mean()
        box_vol_med = v.shift(1).rolling(box).median()
        pump_end_close = c.shift(box + 1)
        pump_start_close = c.shift(box + pump_bars)
        pump_ret = pump_end_close / pump_start_close - 1
        pump_peak = h.shift(box + 1).rolling(pump_bars).max()
        pump_vol_mean = v.shift(box + 1).rolling(pump_bars).mean()
        baseline_vol = v.shift(box + pump_bars + 1).rolling(base_bars).median()
        width = (box_high - box_low) / box_mid
        vol_contraction = box_vol_mean / pump_vol_mean
        pump_vol_ratio = pump_vol_mean / baseline_vol
        retention = (box_low - pump_start_close) / (pump_peak - pump_start_close).replace(0, np.nan)
        breakout_vol = v / box_vol_med
        break_up_strength = c / box_high - 1
        break_dn_strength = box_low / c - 1
        broad_common = (
            (pump_ret >= 0.04) & (pump_vol_ratio >= 1.0) &
            (width <= 0.06) & (vol_contraction <= 1.20) &
            (retention >= 0.20) & (body_ratio >= 0.35) & (breakout_vol >= 1.0)
        )
        long_mask = broad_common & (c > box_high * 1.0005) & (close_loc >= 0.60)
        short_mask = broad_common & (c < box_low * 0.9995) & (close_loc <= 0.40)
        for side, mask, strength in [("long", long_mask, break_up_strength), ("short", short_mask, break_dn_strength)]:
            for i in np.flatnonzero(mask.fillna(False).to_numpy()):
                outcome = first_hit(df, int(i), side, max_hold)
                if outcome is None:
                    continue
                candidates.append({
                    "symbol": symbol, "timeframe": tf, "side": side,
                    "box_minutes": box_minutes, "signal_i": int(i),
                    "signal_time": df.at[i, "timestamp"],
                    "pump_return": float(pump_ret.iat[i]),
                    "pump_volume_ratio": float(pump_vol_ratio.iat[i]),
                    "box_width": float(width.iat[i]),
                    "volume_contraction": float(vol_contraction.iat[i]),
                    "retention": float(retention.iat[i]),
                    "breakout_volume_ratio": float(breakout_vol.iat[i]),
                    "body_ratio": float(body_ratio.iat[i]),
                    "close_location": float(close_loc.iat[i]),
                    "breakout_strength": float(strength.iat[i]),
                    **outcome,
                })
    return pd.DataFrame(candidates)


def split_name(ts: pd.Timestamp) -> str:
    if ts < TRAIN_END: return "train"
    if ts < VALID_END: return "validation"
    if ts < TEST_END: return "test"
    return "outside"


def enforce_nonoverlap(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty: return x
    keep = []
    for _, g in x.sort_values("signal_time").groupby(["symbol", "timeframe"], sort=False):
        last_exit = pd.Timestamp.min.tz_localize("UTC")
        for idx, row in g.iterrows():
            if row["entry_time"] <= last_exit: continue
            keep.append(idx)
            last_exit = row["exit_time"]
    return x.loc[keep].sort_values("signal_time") if keep else x.iloc[0:0]


def metrics(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_net": np.nan, "profit_factor": np.nan,
                "total_compound": np.nan, "max_drawdown": np.nan, "tp_rate": np.nan, "sl_rate": np.nan}
    r = x["net_return"].to_numpy(float)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = np.cumprod(1 + r)
    dd = equity / np.maximum.accumulate(equity) - 1
    return {
        "trades": int(len(x)), "win_rate": float((r > 0).mean()), "avg_net": float(r.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "total_compound": float(equity[-1] - 1), "max_drawdown": float(dd.min()),
        "tp_rate": float((x["reason"] == "tp").mean()), "sl_rate": float((x["reason"] == "sl").mean()),
    }


def parameter_grid() -> Iterable[dict]:
    for pump in [0.06, 0.08, 0.10, 0.12]:
        for width in [0.02, 0.03, 0.04]:
            for bvol in [1.2, 1.5, 2.0]:
                for contraction in [0.70, 0.85, 1.00]:
                    yield {"pump_min": pump, "box_width_max": width,
                           "breakout_volume_min": bvol, "volume_contraction_max": contraction}


def apply_config(cand: pd.DataFrame, cfg: dict, tf: str, side: str, box_minutes: int) -> pd.DataFrame:
    x = cand[(cand["timeframe"] == tf) & (cand["side"] == side) &
             (cand["box_minutes"] == box_minutes) &
             (cand["pump_return"] >= cfg["pump_min"]) &
             (cand["box_width"] <= cfg["box_width_max"]) &
             (cand["breakout_volume_ratio"] >= cfg["breakout_volume_min"]) &
             (cand["volume_contraction"] <= cfg["volume_contraction_max"]) &
             (cand["retention"] >= 0.45) & (cand["body_ratio"] >= 0.55)].copy()
    x = x[x["close_location"] >= 0.72] if side == "long" else x[x["close_location"] <= 0.28]
    return enforce_nonoverlap(x)


def choose_config(cand: pd.DataFrame, tf: str, side: str):
    rows = []
    for box_minutes in [30, 60, 90, 120]:
        for cfg in parameter_grid():
            x = apply_config(cand, cfg, tf, side, box_minutes)
            row = {"timeframe": tf, "side": side, "box_minutes": box_minutes, **cfg}
            for split in ["train", "validation", "test"]:
                for k, val in metrics(x[x["split"] == split]).items(): row[f"{split}_{k}"] = val
            rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[(table["train_trades"] >= 25) & (table["validation_trades"] >= 8) &
                     (table["train_avg_net"] > 0) & (table["validation_avg_net"] > 0)].copy()
    if eligible.empty:
        eligible = table[(table["train_trades"] >= 10) & (table["validation_trades"] >= 3)].copy()
    if eligible.empty: return None, table
    eligible["selection_score"] = (0.35 * eligible["train_avg_net"].fillna(-1) +
                                   0.65 * eligible["validation_avg_net"].fillna(-1) -
                                   0.002 / np.sqrt(eligible["validation_trades"].clip(lower=1)))
    return eligible.sort_values(["selection_score", "validation_trades"], ascending=False).iloc[0].to_dict(), table


def main() -> None:
    all_candidates, data_manifest = [], []
    for tf in TIMEFRAMES:
        for symbol in SYMBOLS:
            df = load_symbol(symbol, tf)
            if df.empty: continue
            data_manifest.append({"symbol": symbol, "timeframe": tf, "bars": len(df),
                                  "start": str(df.timestamp.min()), "end": str(df.timestamp.max())})
            cand = feature_candidates(df, symbol, tf)
            if not cand.empty: all_candidates.append(cand)
            del df
    if not all_candidates: raise RuntimeError("No candidates generated from downloaded market data")
    cand = pd.concat(all_candidates, ignore_index=True)
    cand["split"] = cand["signal_time"].map(split_name)
    cand = cand[cand["split"] != "outside"].copy()
    cand.to_csv(OUT / "all_candidates.csv", index=False)
    config_tables, selections, selected_trades = [], [], []
    for tf in TIMEFRAMES:
        for side in ["long", "short"]:
            best, table = choose_config(cand, tf, side)
            config_tables.append(table)
            if best is None:
                selections.append({"timeframe": tf, "side": side, "status": "no_eligible_config"})
                continue
            cfg = {k: float(best[k]) for k in ["pump_min", "box_width_max", "breakout_volume_min", "volume_contraction_max"]}
            box_minutes = int(best["box_minutes"])
            trades = apply_config(cand, cfg, tf, side, box_minutes)
            selected_trades.append(trades)
            rec = {"timeframe": tf, "side": side, "status": "selected", "box_minutes": box_minutes, **cfg}
            for split in ["train", "validation", "test"]:
                for k, val in metrics(trades[trades["split"] == split]).items(): rec[f"{split}_{k}"] = val
            selections.append(rec)
    pd.concat(config_tables, ignore_index=True).to_csv(OUT / "all_config_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_configs.csv", index=False)
    st = pd.concat(selected_trades, ignore_index=True) if selected_trades else pd.DataFrame()
    st.to_csv(OUT / "selected_trades.csv", index=False)
    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source": "Official Binance USD-M monthly kline archive",
        "symbols_requested": SYMBOLS, "timeframes": TIMEFRAMES,
        "date_range": ["2024-01-01", "2026-06-30"],
        "splits": {"train": "2024-01-01..2025-06-30", "validation": "2025-07-01..2025-12-31", "test": "2026-01-01..2026-06-30"},
        "execution": {"tp": TP, "sl": SL, "round_trip_cost": COST, "entry": "next candle open", "same_bar_rule": "SL first", "max_hold_minutes": 120},
        "manifest": data_manifest, "candidate_count": int(len(cand)), "selections": selections,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
