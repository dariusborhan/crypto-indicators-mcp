"""
Cross-sectional universe analytics: relative-strength ranking, volatility
state, and market breadth.

This module answers questions that coin-by-coin indicators cannot: how is this
asset performing *relative to every other eligible asset*, is its volatility
emerging from an unusually quiet state, and is the market as a whole
broadening or narrowing. It is a candidate-promotion layer only. Nothing here
produces a trade signal, sizes a position, or relaxes any risk gate.

Three design decisions worth stating explicitly.

1. NO PERSISTENCE. The obvious way to compute "is this asset's rank
   improving" is to store each scan's ranks and diff them. That needs a
   database, and on an ephemeral host it silently loses history across
   restarts. Instead, rank as of N days ago is RECOMPUTED from the same
   candle history, using only bars that had closed by that date. This is
   leak-free by construction, survives restarts, and cannot drift out of
   sync with the price data it describes.

2. NO LOOK-AHEAD. Every as-of computation truncates the frame to the bars
   that existed at that time. A metric "as of 3 days ago" never sees bar
   -1 or -2. The currently-forming candle is already excluded upstream by
   datasource.fetch_ohlcv.

3. FAIL LOUD, NEVER ESTIMATE. A symbol whose history is too short for a
   window returns null for that metric and is excluded from that metric's
   ranking. It is never back-filled, interpolated, or scored on a partial
   basis. Fetch failures are returned in a `failures` list, not silently
   dropped, so a scan that covered 20 of 37 assets cannot be mistaken for a
   scan that covered everything.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

import datasource as ds
import indicators as ind

# Daily candles change once a day; a short TTL makes repeated calls within one
# agent run nearly free while still picking up a new daily close promptly.
_CACHE_TTL_SECONDS = 900.0
_MAX_WORKERS = 4

# Windows, in daily bars, used for return and relative-strength measurement.
RETURN_WINDOWS = (1, 3, 7, 30)

# Lookback for "percentile versus this asset's own recent history".
VOL_PERCENTILE_LOOKBACK = 180

# Offsets (in completed daily bars) at which ranks are recomputed.
RANK_OFFSETS = (0, 1, 3)

# Minimum bars needed before an asset is rankable at all.
MIN_BARS_FOR_RANKING = 60

_frame_cache: dict[str, tuple[float, pd.DataFrame]] = {}


# ---------------------------------------------------------------------------
# Universe fetch
# ---------------------------------------------------------------------------

def _fetch_one(symbol: str) -> tuple[str, pd.DataFrame | None, str | None]:
    """Fetch one symbol's completed daily candles. Never raises."""
    now = time.time()
    cached = _frame_cache.get(symbol.upper())
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return symbol.upper(), cached[1], None
    try:
        res = ds.fetch_ohlcv(symbol, "1d", drop_incomplete=True)
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill a scan
        return symbol.upper(), None, f"{type(exc).__name__}: {exc}"
    df = res.df
    if df is None or df.empty:
        return symbol.upper(), None, "Kraken returned no completed candles."
    _frame_cache[symbol.upper()] = (now, df)
    return symbol.upper(), df, None


def fetch_universe(
    symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """
    Fetch completed daily candles for every symbol, concurrently.

    Returns (frames, failures). A symbol that cannot be fetched appears in
    `failures` with its reason and is absent from `frames` -- it is never
    represented by placeholder data.
    """
    wanted: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            wanted.append(u)

    # Warm the shared pair cache on one thread so concurrent workers don't
    # each trigger their own AssetPairs fetch on a cold start.
    try:
        ds.resolve_pair("BTC")
    except Exception:  # noqa: BLE001 - diagnosed properly by the per-symbol fetch
        pass

    frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for sym, df, err in pool.map(_fetch_one, wanted):
            if df is None:
                failures.append({"symbol": sym, "reason": err or "unknown"})
            else:
                frames[sym] = df
    return frames, failures


# ---------------------------------------------------------------------------
# Primitives -- all are "as of" a bar offset, and never look past it
# ---------------------------------------------------------------------------

def _as_of(df: pd.DataFrame, offset: int) -> pd.DataFrame | None:
    """The frame as it stood `offset` completed bars ago."""
    if offset < 0:
        return None
    n = len(df) - offset
    if n < 2:
        return None
    return df.iloc[:n]


def _pct_return(close: pd.Series, bars: int) -> float | None:
    """Percent return over the last `bars` completed bars of this series."""
    if len(close) < bars + 1:
        return None
    a = float(close.iloc[-1 - bars])
    b = float(close.iloc[-1])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (b / a - 1.0) * 100.0


def _realized_vol(close: pd.Series, window: int = 30) -> float | None:
    """Standard deviation of daily log returns, in percent."""
    if len(close) < window + 1:
        return None
    tail = close.iloc[-(window + 1):].astype(float)
    if (tail <= 0).any():
        return None
    rets = np.diff(np.log(tail.to_numpy()))
    if rets.size < 2:
        return None
    sd = float(np.std(rets, ddof=1))
    if not np.isfinite(sd):
        return None
    return sd * 100.0


def _percentile_of_last(series: pd.Series, lookback: int) -> tuple[float | None, int]:
    """
    Where the most recent value sits within its own recent history, 0-100.

    Returns (percentile, sample_size). 100 means the highest reading in the
    lookback window; 0 the lowest.
    """
    s = series.dropna()
    if len(s) < 20:
        return None, len(s)
    window = s.iloc[-lookback:] if len(s) > lookback else s
    current = float(window.iloc[-1])
    if not np.isfinite(current):
        return None, len(window)
    arr = window.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return None, int(arr.size)
    pct = float((arr <= current).sum()) / float(arr.size) * 100.0
    return pct, int(arr.size)


def _rank_percentiles(values: dict[str, float]) -> dict[str, dict[str, Any]]:
    """
    Rank symbols by value, highest first. Percentile 100 = strongest in the
    universe. Only symbols with a finite value are ranked.
    """
    usable = {k: float(v) for k, v in values.items() if v is not None and np.isfinite(v)}
    n = len(usable)
    if n == 0:
        return {}
    ordered = sorted(usable.items(), key=lambda kv: kv[1], reverse=True)
    out: dict[str, dict[str, Any]] = {}
    for i, (sym, val) in enumerate(ordered):
        rank = i + 1
        # Percentile of 100 for the top name, scaling down to ~0 for the last.
        pct = 100.0 * (n - rank) / (n - 1) if n > 1 else 100.0
        out[sym] = {
            "value": ind._round(val),
            "rank": rank,
            "of": n,
            "percentile": ind._round(pct, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Feature A -- cross-sectional relative strength
# ---------------------------------------------------------------------------

def _component_values(
    frames: dict[str, pd.DataFrame], offset: int
) -> dict[str, dict[str, Any]]:
    """
    Raw per-symbol measurements as of `offset` bars ago, before ranking.

    Relative performance is measured against BTC and ETH over matching
    windows, so a coin that rose 5% while BTC rose 8% is correctly treated as
    lagging rather than strong.
    """
    btc = frames.get("BTC")
    eth = frames.get("ETH")

    def bench(df: pd.DataFrame | None, bars: int) -> float | None:
        if df is None:
            return None
        sub = _as_of(df, offset)
        if sub is None:
            return None
        return _pct_return(sub["close"], bars)

    btc_ret = {w: bench(btc, w) for w in RETURN_WINDOWS}
    eth_ret = {w: bench(eth, w) for w in RETURN_WINDOWS}

    out: dict[str, dict[str, Any]] = {}
    for sym, df in frames.items():
        sub = _as_of(df, offset)
        if sub is None or len(sub) < MIN_BARS_FOR_RANKING:
            out[sym] = {
                "rankable": False,
                "reason": (
                    f"{0 if sub is None else len(sub)} completed bars as of this "
                    f"offset; {MIN_BARS_FOR_RANKING} required for ranking."
                ),
            }
            continue
        close = sub["close"]
        rets = {w: _pct_return(close, w) for w in RETURN_WINDOWS}
        vol30 = _realized_vol(close, 30)

        rel_btc = {
            w: (rets[w] - btc_ret[w])
            if rets[w] is not None and btc_ret[w] is not None
            else None
            for w in RETURN_WINDOWS
        }
        rel_eth = {
            w: (rets[w] - eth_ret[w])
            if rets[w] is not None and eth_ret[w] is not None
            else None
            for w in RETURN_WINDOWS
        }
        # Volatility-adjusted momentum: return per unit of its own daily
        # volatility, so a high-beta coin is not rewarded merely for moving.
        vam = (
            rets[7] / vol30
            if rets[7] is not None and vol30 is not None and vol30 > 1e-9
            else None
        )

        out[sym] = {
            "rankable": True,
            "returns_percent": {f"{w}d": ind._round(rets[w], 6) for w in RETURN_WINDOWS},
            "vs_btc_percent": {f"{w}d": ind._round(rel_btc[w], 6) for w in RETURN_WINDOWS},
            "vs_eth_percent": {f"{w}d": ind._round(rel_eth[w], 6) for w in RETURN_WINDOWS},
            "realized_vol_30d_percent": ind._round(vol30, 6),
            "vol_adjusted_momentum_7d": ind._round(vam, 6),
            "_rel_btc_7": rel_btc[7],
            "_rel_btc_30": rel_btc[30],
            "_vam": vam,
        }
    return out


def _composite_percentiles(components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Blend three independent strength measures into one auditable percentile.

    The composite is the mean of the universe percentiles of (a) 7-day
    performance vs BTC, (b) 30-day performance vs BTC, and (c) 7-day
    volatility-adjusted momentum. A symbol missing any of the three is not
    given a composite -- a partial score would not be comparable to a full one.
    """
    rel7 = _rank_percentiles(
        {s: c.get("_rel_btc_7") for s, c in components.items() if c.get("rankable")}
    )
    rel30 = _rank_percentiles(
        {s: c.get("_rel_btc_30") for s, c in components.items() if c.get("rankable")}
    )
    vam = _rank_percentiles(
        {s: c.get("_vam") for s, c in components.items() if c.get("rankable")}
    )

    blended: dict[str, float] = {}
    parts: dict[str, dict[str, Any]] = {}
    for sym in components:
        pieces = []
        detail: dict[str, Any] = {}
        for name, table in (
            ("vs_btc_7d", rel7),
            ("vs_btc_30d", rel30),
            ("vol_adjusted_momentum_7d", vam),
        ):
            entry = table.get(sym)
            detail[name] = entry["percentile"] if entry else None
            if entry is not None:
                pieces.append(entry["percentile"])
        parts[sym] = detail
        if len(pieces) == 3:
            blended[sym] = float(np.mean(pieces))

    final = _rank_percentiles(blended)
    return {"composite": final, "components": parts}


def cross_sectional_strength(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Rank every asset against the rest of the universe, now and at two earlier
    points, and derive rank change and rank acceleration.

    Acceleration is the part worth attention: an asset climbing from the
    bottom half toward the top decile over several days can reflect capital
    rotation that has not yet made the chart look extended.
    """
    per_offset: dict[int, dict[str, Any]] = {}
    for off in RANK_OFFSETS:
        comps = _component_values(frames, off)
        per_offset[off] = {"components": comps, "ranked": _composite_percentiles(comps)}

    now_comps = per_offset[0]["components"]
    now_rank = per_offset[0]["ranked"]["composite"]

    assets: dict[str, Any] = {}
    for sym in sorted(frames):
        comp = now_comps.get(sym, {})
        if not comp.get("rankable"):
            assets[sym] = {
                "rankable": False,
                "reason": comp.get("reason", "not rankable"),
            }
            continue

        pcts: dict[int, float | None] = {}
        for off in RANK_OFFSETS:
            entry = per_offset[off]["ranked"]["composite"].get(sym)
            pcts[off] = entry["percentile"] if entry else None

        p0, p1, p3 = pcts.get(0), pcts.get(1), pcts.get(3)
        rank_change_1d = p0 - p1 if p0 is not None and p1 is not None else None
        rank_change_3d = p0 - p3 if p0 is not None and p3 is not None else None

        # Acceleration compares the most recent day's rank velocity against the
        # average velocity over the two days before it. Positive means the
        # climb is steepening, not merely continuing.
        acceleration = None
        if p0 is not None and p1 is not None and p3 is not None:
            recent_velocity = p0 - p1
            prior_velocity = (p1 - p3) / 2.0
            acceleration = recent_velocity - prior_velocity

        entry_now = now_rank.get(sym)
        public = {k: v for k, v in comp.items() if not k.startswith("_")}
        public.update(
            {
                "composite_percentile": entry_now["percentile"] if entry_now else None,
                "composite_rank": entry_now["rank"] if entry_now else None,
                "ranked_against": entry_now["of"] if entry_now else None,
                "component_percentiles": per_offset[0]["ranked"]["components"].get(sym),
                "percentile_1d_ago": ind._round(p1, 4) if p1 is not None else None,
                "percentile_3d_ago": ind._round(p3, 4) if p3 is not None else None,
                "rank_change_1d": ind._round(rank_change_1d, 4),
                "rank_change_3d": ind._round(rank_change_3d, 4),
                "rank_acceleration": ind._round(acceleration, 4),
            }
        )
        assets[sym] = public

    return {
        "assets": assets,
        "method": {
            "composite": (
                "Mean of the universe percentiles of three independent measures: "
                "7-day return vs BTC, 30-day return vs BTC, and 7-day return "
                "divided by 30-day realized volatility. Re-ranked to a percentile."
            ),
            "percentile_meaning": "100 = strongest in the eligible universe.",
            "as_of_offsets_in_daily_bars": list(RANK_OFFSETS),
            "historical_ranks": (
                "Recomputed from candle history truncated to the bars that had "
                "closed at that time, not read from stored state."
            ),
            "return_windows_days": list(RETURN_WINDOWS),
            "min_bars_for_ranking": MIN_BARS_FOR_RANKING,
        },
    }


# ---------------------------------------------------------------------------
# Feature B -- volatility compression and expansion
# ---------------------------------------------------------------------------

def _bandwidth_series(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper, lower = mid + std * sd, mid - std * sd
    return (upper - lower) / mid.replace(0, np.nan) * 100.0


def _atr_percent_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr / close.replace(0, np.nan) * 100.0


def volatility_state(df: pd.DataFrame) -> dict[str, Any]:
    """
    Where this asset's volatility sits within its own recent history, and
    whether a compressed state has begun expanding.

    A compression reading is not a direction. A band squeeze precedes a move
    either way; this only promotes the asset for a closer look.
    """
    if len(df) < 60:
        return {
            "available": False,
            "reason": f"{len(df)} completed daily bars; 60 required for volatility state.",
        }

    close = df["close"]
    bw = _bandwidth_series(close)
    atrp = _atr_percent_series(df)

    bw_pct, bw_n = _percentile_of_last(bw, VOL_PERCENTILE_LOOKBACK)
    atr_pct, atr_n = _percentile_of_last(atrp, VOL_PERCENTILE_LOOKBACK)

    # Range contraction: the recent 10-bar range against the 40-bar range.
    # Well below 1 means price has coiled relative to its own recent swing.
    hi10, lo10 = float(df["high"].iloc[-10:].max()), float(df["low"].iloc[-10:].min())
    hi40, lo40 = float(df["high"].iloc[-40:].max()), float(df["low"].iloc[-40:].min())
    range10 = hi10 - lo10
    range40 = hi40 - lo40
    contraction = range10 / range40 if range40 > 1e-12 else None
    position_in_range = (
        (float(close.iloc[-1]) - lo10) / range10 if range10 > 1e-12 else None
    )

    vol5 = float(df["volume"].iloc[-5:].mean())
    vol20 = float(df["volume"].iloc[-20:].mean())
    volume_ratio = vol5 / vol20 if vol20 > 1e-12 else None

    # Was this asset genuinely quiet recently? Look for a compressed bandwidth
    # reading within the last 10 bars, not merely a quiet reading today.
    recent_bw = bw.dropna().iloc[-10:]
    hist_bw = bw.dropna().iloc[-VOL_PERCENTILE_LOOKBACK:]
    was_compressed = False
    if len(hist_bw) >= 20 and len(recent_bw) > 0:
        q25 = float(np.percentile(hist_bw.to_numpy(dtype=float), 25))
        was_compressed = bool((recent_bw <= q25).any())

    expanding = False
    expansion_ratio = None
    atr_expanding = False
    if len(bw.dropna()) >= 6:
        b = bw.dropna()
        baseline = float(b.iloc[-6:-1].min())
        current_bw = float(b.iloc[-1])
        expansion_ratio = current_bw / baseline if baseline > 1e-12 else None
        # Keep the release detector sensitive; the magnitude is exposed
        # separately so candidate priority can distinguish weak from strong releases.
        expanding = bool(float(b.iloc[-1]) > float(b.iloc[-4]))
    if len(atrp.dropna()) >= 4:
        a = atrp.dropna()
        atr_expanding = bool(float(a.iloc[-1]) > float(a.iloc[-4]))

    transition = bool(
        was_compressed
        and expanding
        and volume_ratio is not None
        and volume_ratio >= 1.1
    )


    return {
        "available": True,
        "bollinger_bandwidth_percent": ind._round(
            float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else None, 6
        ),
        "bollinger_bandwidth_percentile": ind._round(bw_pct, 4),
        "bandwidth_sample_size": bw_n,
        "atr_percent_of_price": ind._round(
            float(atrp.iloc[-1]) if pd.notna(atrp.iloc[-1]) else None, 6
        ),
        "atr_percentile": ind._round(atr_pct, 4),
        "atr_sample_size": atr_n,
        "range_contraction_10_vs_40": ind._round(contraction, 4),
        "position_in_10d_range": ind._round(position_in_range, 4),
        "volume_5d_vs_20d": ind._round(volume_ratio, 4),
        "was_compressed_within_10_bars": was_compressed,
        "bandwidth_expanding": expanding,
        "bandwidth_expansion_ratio_from_5d_min": ind._round(expansion_ratio, 4),
        "atr_expanding": atr_expanding,
        "compression_expansion_transition": transition,
        "lookback_bars_for_percentiles": min(VOL_PERCENTILE_LOOKBACK, len(df)),
        "note": (
            "A squeeze or its release is not directional. Transition=true means "
            "a historically quiet asset has begun expanding on rising volume, "
            "which warrants a deep dive, never an entry on its own."
        ),
    }


# ---------------------------------------------------------------------------
# Feature C -- market breadth and rotation
# ---------------------------------------------------------------------------

def _breadth_at(frames: dict[str, pd.DataFrame], offset: int) -> dict[str, Any] | None:
    btc = frames.get("BTC")
    btc_sub = _as_of(btc, offset) if btc is not None else None
    btc_7 = _pct_return(btc_sub["close"], 7) if btc_sub is not None else None

    above20 = above50 = pos7 = beat_btc = 0
    counted20 = counted50 = counted7 = counted_btc = 0
    rsis: list[float] = []
    rets: list[float] = []

    for sym, df in frames.items():
        sub = _as_of(df, offset)
        if sub is None or len(sub) < 50:
            continue
        close = sub["close"]
        last = float(close.iloc[-1])

        s20 = ind.sma(close, 20)
        if s20.available:
            counted20 += 1
            above20 += int(last > float(s20.value))
        s50 = ind.sma(close, 50)
        if s50.available:
            counted50 += 1
            above50 += int(last > float(s50.value))

        r7 = _pct_return(close, 7)
        if r7 is not None:
            counted7 += 1
            rets.append(r7)
            pos7 += int(r7 > 0)
            if btc_7 is not None:
                counted_btc += 1
                beat_btc += int(r7 > btc_7)

        r = ind.rsi(close, 14)
        if r.available:
            rsis.append(float(r.value))

    if counted20 == 0 or counted7 == 0:
        return None

    def pct(num: int, den: int) -> float | None:
        return ind._round(100.0 * num / den, 4) if den else None

    return {
        "assets_measured": len(frames),
        "percent_above_sma20": pct(above20, counted20),
        "percent_above_sma50": pct(above50, counted50),
        "percent_positive_7d": pct(pos7, counted7),
        "percent_outperforming_btc_7d": pct(beat_btc, counted_btc),
        "median_rsi14": ind._round(float(np.median(rsis)), 4) if rsis else None,
        "median_return_7d_percent": ind._round(float(np.median(rets)), 4) if rets else None,
        "sample_sizes": {
            "sma20": counted20,
            "sma50": counted50,
            "return_7d": counted7,
            "vs_btc": counted_btc,
            "rsi": len(rsis),
        },
    }


def market_breadth(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    A market-level snapshot built from the eligible universe itself, plus the
    one-day change in each measure, so a BTC-led move can be told apart from
    broad participation.
    """
    now = _breadth_at(frames, 0)
    prior = _breadth_at(frames, 1)
    if now is None:
        return {
            "available": False,
            "reason": "Not enough assets with sufficient history to measure breadth.",
        }

    deltas: dict[str, Any] = {}
    if prior is not None:
        for key in (
            "percent_above_sma20",
            "percent_above_sma50",
            "percent_positive_7d",
            "percent_outperforming_btc_7d",
            "median_rsi14",
            "median_return_7d_percent",
        ):
            a, b = now.get(key), prior.get(key)
            deltas[key] = ind._round(a - b, 4) if a is not None and b is not None else None

    # Explicit numeric rules, so the label cannot drift with narrative framing.
    a20 = now.get("percent_above_sma20")
    d20 = deltas.get("percent_above_sma20")
    if a20 is None:
        classification, rule = "unknown", "percent_above_sma20 unavailable"
    elif d20 is not None and d20 >= 5.0 and a20 >= 50.0:
        classification = "broadening"
        rule = "percent_above_sma20 >= 50 and rose >= 5 points in a day"
    elif d20 is not None and d20 <= -5.0:
        classification = "narrowing"
        rule = "percent_above_sma20 fell >= 5 points in a day"
    elif a20 >= 60.0:
        classification, rule = "broad", "percent_above_sma20 >= 60"
    elif a20 <= 35.0:
        classification, rule = "weak", "percent_above_sma20 <= 35"
    else:
        classification, rule = "stable", "no threshold crossed"

    rotation = None
    obtc = now.get("percent_outperforming_btc_7d")
    if obtc is not None:
        if obtc >= 55.0:
            rotation = "altcoin-led (majority outperforming BTC over 7d)"
        elif obtc <= 30.0:
            rotation = "BTC-led (most alts lagging BTC over 7d)"
        else:
            rotation = "mixed"

    return {
        "available": True,
        "current": now,
        "one_day_change": deltas or None,
        "classification": classification,
        "classification_rule": rule,
        "rotation": rotation,
        "note": (
            "Breadth is context for candidate selection. It never overrides an "
            "entry, reward-to-risk, liquidity, correlation, or risk-budget gate."
        ),
    }



# ---------------------------------------------------------------------------
# Feature D -- anomaly and correlation-regime intelligence
# ---------------------------------------------------------------------------

def _zscore_last(series: pd.Series, lookback: int = 90, min_samples: int = 30) -> tuple[float | None, int]:
    """Z-score the latest finite observation against PRIOR observations only."""
    x = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < min_samples + 1:
        return None, max(0, len(x) - 1)
    hist = x.iloc[-(lookback + 1):-1] if len(x) > lookback else x.iloc[:-1]
    if len(hist) < min_samples:
        return None, len(hist)
    sd = float(hist.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None, len(hist)
    return float((x.iloc[-1] - hist.mean()) / sd), len(hist)


def anomaly_state(df: pd.DataFrame) -> dict[str, Any]:
    """Return/volume/range anomalies versus the asset's own prior history."""
    if len(df) < 40:
        return {"available": False, "reason": f"{len(df)} bars; 40 required."}
    close = df["close"].astype(float)
    ret = close.pct_change() * 100.0
    logvol = np.log1p(df["volume"].astype(float))
    range_pct = (df["high"].astype(float) - df["low"].astype(float)) / close.replace(0, np.nan) * 100.0
    rz, rn = _zscore_last(ret)
    vz, vn = _zscore_last(logvol)
    gz, gn = _zscore_last(range_pct)
    return {
        "available": any(v is not None for v in (rz, vz, gz)),
        "return_1d_zscore": ind._round(rz, 4),
        "volume_log_zscore": ind._round(vz, 4),
        "range_zscore": ind._round(gz, 4),
        "sample_sizes": {"return": rn, "volume": vn, "range": gn},
        "lookback_bars": 90,
        "note": "Z-scores use prior observations only; the current bar is never included in its own baseline.",
    }


def correlation_regime(df: pd.DataFrame, btc: pd.DataFrame) -> dict[str, Any]:
    """Short-vs-medium BTC correlation and the resulting correlation delta."""
    a = df["close"].astype(float).pct_change().dropna()
    b = btc["close"].astype(float).pct_change().dropna()
    aligned = pd.concat([a.rename("asset"), b.rename("btc")], axis=1).dropna()
    if len(aligned) < 61:
        return {"available": False, "reason": f"{len(aligned)} aligned returns; 61 required."}
    short = float(aligned.iloc[-20:].corr().iloc[0, 1])
    medium = float(aligned.iloc[-60:].corr().iloc[0, 1])
    return {
        "available": True,
        "correlation_to_btc_20d": ind._round(short, 4),
        "correlation_to_btc_60d": ind._round(medium, 4),
        "correlation_delta_20d_vs_60d": ind._round(short - medium, 4),
        "decoupling": bool(short <= medium - 0.20),
        "note": "Decoupling is supporting evidence only; it is not directional by itself.",
    }


def _candidate_priority(strength: dict[str, Any], vol: dict[str, Any], anomaly: dict[str, Any], corr: dict[str, Any]) -> dict[str, Any]:
    """Transparent 0-100 interest score used ONLY to prioritize deep dives."""
    p = strength.get("composite_percentile")
    leadership = float(p) if p is not None else 0.0
    accel = max(0.0, float(strength.get("rank_acceleration") or 0.0))
    chg3 = max(0.0, float(strength.get("rank_change_3d") or 0.0))
    emergence = min(100.0, 4.0 * accel + 2.0 * chg3)
    vz = anomaly.get("volume_log_zscore") if anomaly.get("available") else None
    participation = min(100.0, max(0.0, 50.0 + 20.0 * float(vz))) if vz is not None else 0.0
    bw = vol.get("bollinger_bandwidth_percentile") if vol.get("available") else None
    vol_transition = 85.0 if vol.get("compression_expansion_transition") else (max(0.0, 60.0 - float(bw)) if bw is not None else 0.0)
    delta = corr.get("correlation_delta_20d_vs_60d") if corr.get("available") else None
    independence = min(100.0, max(0.0, -float(delta) * 200.0)) if delta is not None else 0.0
    dims = {
        "leadership": leadership, "emergence": emergence, "participation": participation,
        "volatility_transition": vol_transition, "independence": independence,
    }
    # Leadership matters most, but an established leader can remain interesting
    # even when rank acceleration has naturally flattened.
    score = 0.35*leadership + 0.25*emergence + 0.15*participation + 0.15*vol_transition + 0.10*independence
    return {"score": ind._round(score, 4), "dimensions": {k: ind._round(v, 4) for k,v in dims.items()}}

# ---------------------------------------------------------------------------
# Candidate promotion
# ---------------------------------------------------------------------------
#
# Two lessons from the first live scan, which promoted 19 of 35 assets --
# including two in the bottom sixth of the universe -- when the intent was a
# handful of the strongest:
#
#   1. RELATIVE STRENGTH IS A GATE, NOT A VOTE. Scoring "percentile >= 70" as
#      a category guarantees that 30% of the universe fires it on every scan,
#      by definition, because a percentile is a rank. It now gates promotion
#      instead: nothing below the universe median can be promoted at all. A
#      long-only account should not spend a deep dive on the weakest names.
#
#   2. THRESHOLDS MUST BE RELATIVE TO THE SCAN, NOT ABSOLUTE. On a day when
#      the whole market is expanding, an absolute rule like "volume >= 1.4x"
#      fires on half the universe and carries no information. Two defences:
#      the volume bar is set from this scan's own distribution, and any
#      category firing on more than a set fraction of the universe is
#      suppressed entirely for that scan and reported as non-discriminating.
#      A signal shared by half the market is a description of the market, not
#      a reason to single an asset out.

# An asset below this universe percentile is never promoted, whatever else
# fires. A genuine rotation crosses the median quickly, and the universe is
# scanned three times a day, so this still catches it early.
MIN_PROMOTION_PERCENTILE = 50.0

# A category firing on more than this fraction of the universe is suppressed
# for the scan: it is describing the regime, not distinguishing an asset.
COMMON_SIGNAL_FRACTION = 0.35

# Independent categories that must agree before an asset is promoted.
REQUIRED_CATEGORIES = 2

# Hard cap on the shortlist, ranked by composite percentile.
MAX_PROMOTED = 6

ACCELERATION_THRESHOLD = 8.0
RANK_CHANGE_THRESHOLD = 10.0

# The volume bar is the higher of this scan's upper quartile and this floor,
# so a flat market cannot make an ordinary 1.05x reading look unusual.
VOLUME_PERCENTILE_CUTOFF = 75.0
VOLUME_ABSOLUTE_FLOOR = 1.2


def _signal_flags(
    strength: dict[str, Any], vol: dict[str, Any], volume_bar: float
) -> tuple[dict[str, bool], dict[str, str]]:
    """Raw category flags for one asset, before universe-wide suppression."""
    flags: dict[str, bool] = {}
    why: dict[str, str] = {}
    if not strength.get("rankable"):
        return flags, why

    accel = strength.get("rank_acceleration")
    chg3 = strength.get("rank_change_3d")
    if accel is not None and accel >= ACCELERATION_THRESHOLD:
        flags["rank_trajectory"] = True
        why["rank_trajectory"] = f"rank acceleration +{accel:.0f} (climb is steepening)"
    elif chg3 is not None and chg3 >= RANK_CHANGE_THRESHOLD:
        flags["rank_trajectory"] = True
        why["rank_trajectory"] = f"rank improved {chg3:.0f} percentile points over 3 days"

    if vol.get("available"):
        if vol.get("compression_expansion_transition"):
            flags["volatility_state"] = True
            why["volatility_state"] = (
                "volatility expanding out of a compressed state on rising volume"
            )
        vr = vol.get("volume_5d_vs_20d")
        if vr is not None and vr >= volume_bar:
            flags["volume"] = True
            why["volume"] = (
                f"5-day volume {vr:.2f}x its 20-day average "
                f"(scan bar {volume_bar:.2f}x)"
            )
    return flags, why


def _volume_bar(vol_states: dict[str, dict[str, Any]]) -> float:
    """The volume ratio an asset must beat, set from this scan's spread."""
    ratios = [
        float(v["volume_5d_vs_20d"])
        for v in vol_states.values()
        if v.get("available") and v.get("volume_5d_vs_20d") is not None
    ]
    if len(ratios) < 8:
        return VOLUME_ABSOLUTE_FLOOR
    return max(
        float(np.percentile(np.asarray(ratios, dtype=float), VOLUME_PERCENTILE_CUTOFF)),
        VOLUME_ABSOLUTE_FLOOR,
    )


def _suppressed_categories(
    raw: dict[str, dict[str, bool]], universe_size: int
) -> dict[str, dict[str, Any]]:
    """Categories so widely fired this scan that they no longer discriminate."""
    out: dict[str, dict[str, Any]] = {}
    if universe_size <= 0:
        return out
    for cat in ("rank_trajectory", "volatility_state", "volume"):
        fired = sum(1 for f in raw.values() if f.get(cat))
        fraction = fired / universe_size
        if fraction > COMMON_SIGNAL_FRACTION:
            out[cat] = {
                "fired_on": fired,
                "of": universe_size,
                "fraction": ind._round(fraction, 3),
                "reason": (
                    "Fired on more than "
                    f"{COMMON_SIGNAL_FRACTION:.0%} of the universe, so it describes "
                    "current market conditions rather than distinguishing an asset. "
                    "Ignored for promotion this scan."
                ),
            }
    return out



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scan_universe(
    symbols: list[str], detail: str = "summary", max_symbols: int = 120
) -> dict[str, Any]:
    """
    Run the full cross-sectional pass over an eligible universe.

    detail="summary" (the default) returns the breadth snapshot, the promoted
    shortlist with its reasons, and a one-line-per-asset leaderboard. That is
    what an agent needs to decide where to deep-dive, and it stays small
    enough to read mid-run. detail="full" adds every per-asset measurement,
    which runs to tens of thousands of tokens on a real universe -- use it for
    inspection, not inside a scheduled run.

    max_symbols defaults to 120 -- comfortably above the ~90-symbol size of a
    typical brokerage's full crypto catalog, so a caller building "the entire
    permitted universe minus exclusions" in one pass does not get truncated
    or rejected mid-run. Fetches are threaded (_MAX_WORKERS=4) and each hits
    only Kraken's public, unauthenticated OHLC endpoint, so the wall-clock
    cost of a larger universe is longer scan time, not rate-limit risk. The
    cap still exists to catch a caller accidentally passing an unfiltered
    exchange-wide symbol list.
    """
    if detail not in ("summary", "full"):
        return {"error": "detail must be 'summary' or 'full'."}
    if not symbols:
        return {"error": "No symbols supplied. Pass the eligible universe."}
    if len(symbols) > max_symbols:
        return {
            "error": (
                f"{len(symbols)} symbols requested; this tool accepts at most "
                f"{max_symbols} per call to stay within Kraken rate limits. "
                f"Split the universe across calls."
            )
        }

    started = time.time()
    frames, failures = fetch_universe(symbols)

    # BTC anchors every relative measurement; without it nothing is comparable.
    if "BTC" not in frames:
        return {
            "error": (
                "BTC candles could not be fetched, so relative-strength and "
                "breadth cannot be computed. No partial results are returned."
            ),
            "fetch_failures": failures,
        }

    strength = cross_sectional_strength(frames)
    breadth = market_breadth(frames)
    vol_states = {sym: volatility_state(df) for sym, df in frames.items()}
    anomaly_states = {sym: anomaly_state(df) for sym, df in frames.items()}
    corr_states = {sym: correlation_regime(df, frames["BTC"]) for sym, df in frames.items()}
    priority = {sym: _candidate_priority(strength["assets"].get(sym, {}), vol_states[sym], anomaly_states[sym], corr_states[sym]) for sym in frames}

    # Pass 1: raw flags for every asset, against a volume bar drawn from this
    # scan's own distribution rather than a fixed ratio.
    volume_bar = _volume_bar(vol_states)
    raw_flags: dict[str, dict[str, bool]] = {}
    raw_why: dict[str, dict[str, str]] = {}
    for sym in frames:
        f, w = _signal_flags(strength["assets"].get(sym, {}), vol_states[sym], volume_bar)
        raw_flags[sym], raw_why[sym] = f, w

    # Pass 2: drop categories that fired so widely they describe the regime.
    suppressed = _suppressed_categories(raw_flags, len(frames))

    # Pass 3: apply the relative-strength gate and the agreement requirement.
    candidates: list[dict[str, Any]] = []
    promotion_by_symbol: dict[str, dict[str, Any]] = {}
    for sym in frames:
        s = strength["assets"].get(sym, {})
        pctl = s.get("composite_percentile")
        surviving = sorted(c for c in raw_flags[sym] if c not in suppressed)
        gate_ok = pctl is not None and pctl >= MIN_PROMOTION_PERCENTILE

        blocked = None
        if not s.get("rankable"):
            blocked = "not rankable"
        elif not gate_ok:
            blocked = (
                f"below the {MIN_PROMOTION_PERCENTILE:.0f}th relative-strength "
                f"percentile (at {pctl:.0f})" if pctl is not None else "no composite score"
            )
        elif len(surviving) < REQUIRED_CATEGORIES:
            blocked = (
                f"{len(surviving)} discriminating category(ies); "
                f"{REQUIRED_CATEGORIES} required"
            )

        reasons = [raw_why[sym][c] for c in surviving]
        # The pattern worth calling out: strength and unusual volume building
        # before the price move has become extreme. A highlight, not a vote --
        # it derives from volume, so it does not count toward agreement.
        r7 = (s.get("returns_percent") or {}).get("7d")
        if (
            "volume" in surviving
            and pctl is not None
            and pctl >= 60.0
            and r7 is not None
            and abs(r7) < 10.0
        ):
            reasons.append(
                "unusual volume and firm relative strength without an extreme "
                "price move yet"
            )

        promo = {
            "promote": blocked is None,
            "categories": surviving,
            "reasons": reasons,
            "blocked_by": blocked,
        }
        promotion_by_symbol[sym] = promo
        if blocked is None:
            candidates.append(
                {
                    "symbol": sym,
                    "composite_percentile": pctl,
                    "rank_acceleration": s.get("rank_acceleration"),
                    "categories": surviving,
                    "reasons": reasons,
                }
            )

    candidates.sort(
        key=lambda d: d["composite_percentile"] if d["composite_percentile"] is not None else -1,
        reverse=True,
    )
    over_cap = max(0, len(candidates) - MAX_PROMOTED)
    promoted = candidates[:MAX_PROMOTED]

    leaderboard = []
    for sym in frames:
        s = strength["assets"].get(sym, {})
        v = vol_states[sym]
        leaderboard.append(
            {
                "symbol": sym,
                "percentile": s.get("composite_percentile"),
                "return_7d": (s.get("returns_percent") or {}).get("7d"),
                "vs_btc_7d": (s.get("vs_btc_percent") or {}).get("7d"),
                "rank_change_3d": s.get("rank_change_3d"),
                "rank_acceleration": s.get("rank_acceleration"),
                "volume_5d_vs_20d": v.get("volume_5d_vs_20d") if v.get("available") else None,
                "vol_transition": v.get("compression_expansion_transition")
                if v.get("available")
                else None,
                "promoted": promotion_by_symbol[sym]["promote"],
                "candidate_priority": priority[sym]["score"],
                "return_1d_zscore": anomaly_states[sym].get("return_1d_zscore"),
                "volume_zscore": anomaly_states[sym].get("volume_log_zscore"),
                "btc_corr_20d": corr_states[sym].get("correlation_to_btc_20d"),
                "btc_corr_delta": corr_states[sym].get("correlation_delta_20d_vs_60d"),
            }
        )
    leaderboard.sort(key=lambda d: d["candidate_priority"], reverse=True)

    # Always provide a small deterministic deep-dive queue on a healthy scan.
    # This prevents a language model from declaring the entire universe boring.
    # It does NOT force an entry: every downstream technical/risk gate remains.
    deep_dive_candidates = [
        {"symbol": row["symbol"], "candidate_priority": row["candidate_priority"],
         "percentile": row["percentile"], "promoted": row["promoted"]}
        for row in leaderboard if row["symbol"] != "BTC"
    ][:3]

    result: dict[str, Any] = {
        "source": "Kraken public API, completed daily candles only",
        "detail": detail,
        "requested": len(symbols),
        "fetched": len(frames),
        "fetch_failures": failures,
        "elapsed_seconds": ind._round(time.time() - started, 4),
        "market_breadth": breadth,
        "promoted_candidates": promoted,
        "recommended_deep_dives": deep_dive_candidates,
        "deep_dive_policy": "Deep-dive the top 3 non-BTC quantitative candidates on every healthy full scan. This prioritizes investigation only and never forces a trade.",
        "promotion_rules": {
            "gate": (
                f"Relative-strength percentile must be at least "
                f"{MIN_PROMOTION_PERCENTILE:.0f}. Strength gates promotion; it is "
                f"not itself a category, because a percentile threshold would fire "
                f"on a fixed share of the universe every scan."
            ),
            "requirement": (
                f"At least {REQUIRED_CATEGORIES} discriminating categories must "
                f"agree: rank_trajectory, volatility_state, volume."
            ),
            "volume_bar_this_scan": ind._round(volume_bar, 4),
            "suppressed_categories": suppressed or None,
            "cap": MAX_PROMOTED,
            "candidates_over_cap": over_cap,
        },
        "leaderboard": leaderboard,
        "method": strength["method"],
        "data_limitations": [
            "Kraken spot only. No funding rates, open interest, or futures positioning.",
            "No on-chain data and no news. Nothing here reflects fundamentals.",
            "Promotion selects candidates for deep-dive analysis. It is not a buy "
            "signal and does not relax any existing entry, reward-to-risk, "
            "liquidity, correlation, or risk-budget gate.",
            "Ranks are relative to the supplied universe only; a top percentile in "
            "a weak market is still a weak asset in absolute terms.",
        ],
    }

    if detail == "full":
        result["assets"] = {
            sym: {
                "bars_available": len(frames[sym]),
                "relative_strength": strength["assets"].get(sym, {}),
                "volatility_state": vol_states[sym],
                "anomaly_state": anomaly_states[sym],
                "correlation_regime": corr_states[sym],
                "candidate_priority": priority[sym],
                "promotion": promotion_by_symbol[sym],
            }
            for sym in sorted(frames)
        }
    return result
