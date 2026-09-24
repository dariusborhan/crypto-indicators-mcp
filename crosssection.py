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
    if len(bw.dropna()) >= 4:
        b = bw.dropna()
        expanding = bool(float(b.iloc[-1]) > float(b.iloc[-4]))

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
# Candidate promotion
# ---------------------------------------------------------------------------

# Promotion requires agreement across independent categories, so no single
# measure can push an asset forward on its own.
STRONG_PERCENTILE = 70.0
ACCELERATION_THRESHOLD = 8.0
RANK_CHANGE_THRESHOLD = 10.0
VOLUME_ANOMALY_RATIO = 1.4


def _promotion_for(
    sym: str, strength: dict[str, Any], vol: dict[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    categories: set[str] = set()

    if not strength.get("rankable"):
        return {"promote": False, "categories_met": 0, "reasons": [], "eligible": False}

    pctl = strength.get("composite_percentile")
    accel = strength.get("rank_acceleration")
    chg3 = strength.get("rank_change_3d")

    if pctl is not None and pctl >= STRONG_PERCENTILE:
        reasons.append(f"relative-strength percentile {pctl:.0f} (top of universe)")
        categories.add("relative_strength")
    if accel is not None and accel >= ACCELERATION_THRESHOLD:
        reasons.append(f"rank acceleration +{accel:.0f} (climb is steepening)")
        categories.add("rank_trajectory")
    elif chg3 is not None and chg3 >= RANK_CHANGE_THRESHOLD:
        reasons.append(f"rank improved {chg3:.0f} percentile points over 3 days")
        categories.add("rank_trajectory")

    if vol.get("available"):
        if vol.get("compression_expansion_transition"):
            reasons.append("volatility expanding out of a compressed state on rising volume")
            categories.add("volatility_state")
        vr = vol.get("volume_5d_vs_20d")
        if vr is not None and vr >= VOLUME_ANOMALY_RATIO:
            reasons.append(f"5-day volume {vr:.2f}x its 20-day average")
            categories.add("volume")

    # The pattern the spec singles out: strength building without the price
    # move having become extreme yet.
    r7 = (strength.get("returns_percent") or {}).get("7d")
    if (
        "volume" in categories
        and pctl is not None
        and pctl >= 60.0
        and r7 is not None
        and abs(r7) < 10.0
    ):
        reasons.append(
            "unusual volume and firm relative strength without an extreme price move yet"
        )
        categories.add("early_signature")

    return {
        "promote": len(categories) >= 2,
        "categories_met": len(categories),
        "categories": sorted(categories),
        "reasons": reasons,
        "eligible": True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scan_universe(symbols: list[str], max_symbols: int = 60) -> dict[str, Any]:
    """
    Run the full cross-sectional pass over an eligible universe.

    Returns per-asset relative strength, rank trajectory, volatility state and
    promotion reasons, plus a market-level breadth snapshot.
    """
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

    assets: dict[str, Any] = {}
    promoted: list[dict[str, Any]] = []
    for sym in sorted(frames):
        s = strength["assets"].get(sym, {})
        v = volatility_state(frames[sym])
        promo = _promotion_for(sym, s, v)
        assets[sym] = {
            "bars_available": len(frames[sym]),
            "relative_strength": s,
            "volatility_state": v,
            "promotion": promo,
        }
        if promo.get("promote"):
            promoted.append(
                {
                    "symbol": sym,
                    "composite_percentile": s.get("composite_percentile"),
                    "rank_acceleration": s.get("rank_acceleration"),
                    "categories": promo.get("categories"),
                    "reasons": promo.get("reasons"),
                }
            )

    promoted.sort(
        key=lambda d: (
            d.get("composite_percentile") if d.get("composite_percentile") is not None else -1
        ),
        reverse=True,
    )

    return {
        "source": "Kraken public API, completed daily candles only",
        "requested": len(symbols),
        "fetched": len(frames),
        "fetch_failures": failures,
        "elapsed_seconds": ind._round(time.time() - started, 4),
        "market_breadth": breadth,
        "promoted_candidates": promoted,
        "promotion_rules": {
            "requirement": "At least two independent categories must agree.",
            "categories": [
                "relative_strength",
                "rank_trajectory",
                "volatility_state",
                "volume",
                "early_signature",
            ],
            "thresholds": {
                "strong_percentile": STRONG_PERCENTILE,
                "rank_acceleration": ACCELERATION_THRESHOLD,
                "rank_change_3d": RANK_CHANGE_THRESHOLD,
                "volume_ratio_5d_vs_20d": VOLUME_ANOMALY_RATIO,
            },
        },
        "assets": assets,
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
