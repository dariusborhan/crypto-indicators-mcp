"""
Technical indicator computation.

Design principle (from the trading strategy spec):
    "If the data needed to compute a given indicator reliably isn't available,
     say so explicitly rather than approximating or estimating a value."

Every function here returns either a real value computed from real data, or
None together with a machine-readable reason. Nothing is ever extrapolated,
padded, or guessed. Callers surface the None to the agent so it can decline to
trade rather than reason over a fabricated number.

All formulas are implemented directly in pandas rather than via a TA library,
so the math is auditable and there is no dependency drift. Correctness is
verified in test_indicators.py against textbook loop implementations and
mathematical invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Warm-up requirements
# ---------------------------------------------------------------------------
# Wilder-smoothed indicators (RSI, ATR) are recursive: the first value seeds an
# exponential average that only converges after several multiples of the
# period. Reporting an RSI computed from exactly `period` bars is technically
# "a number" but is dominated by the seed value and is not comparable to the
# RSI any charting platform would show. We therefore distinguish:
#
#   bars < MIN_BARS          -> None, insufficient data
#   MIN_BARS <= bars < WARM  -> value returned, warmup_sufficient=False
#   bars >= WARM             -> value returned, warmup_sufficient=True
#
# Rather than a blanket "4x the period" rule of thumb, the requirement is
# derived from how fast the seed value actually decays out of the average.
#
# Both EMA and Wilder smoothing are of the form V_t = a*P_t + (1-a)*V_{t-1},
# seeded with the first observation. After `n` bars the seed still carries
# weight (1-a)^n. We call the average converged once that weight falls below
# SEED_WEIGHT_THRESHOLD, and solve for the n at which that happens.
#
# This matters because the two families have very different alphas:
#   EMA(200):    a = 2/201 = 0.00995  -> converged after ~461 bars
#   Wilder(14):  a = 1/14  = 0.0714   -> converged after ~63 bars
#
# The old 4x rule demanded 800 bars for a 200-period EMA. Most exchanges cap a
# single request well below that (Kraken returns 720), so EMA(200) would have
# been flagged provisional forever despite the seed contributing under 0.1%.
SEED_WEIGHT_THRESHOLD = 0.01


def _bars_to_converge(alpha: float) -> int:
    """Bars until the seed's weight drops below SEED_WEIGHT_THRESHOLD."""
    if alpha <= 0.0 or alpha >= 1.0:
        return 1
    return int(np.ceil(np.log(SEED_WEIGHT_THRESHOLD) / np.log(1.0 - alpha)))


def _ema_warmup(period: int) -> int:
    """Bars for a span-`period` EMA (alpha = 2/(period+1)) to converge."""
    return _bars_to_converge(2.0 / (period + 1.0))


def _wilder_warmup(period: int) -> int:
    """Bars for Wilder smoothing (alpha = 1/period) to converge."""
    return _bars_to_converge(1.0 / period)


@dataclass
class IndicatorResult:
    """A single indicator value plus the provenance needed to trust it."""

    value: Any = None
    available: bool = False
    reason: str | None = None
    warmup_sufficient: bool | None = None
    bars_used: int | None = None
    bars_required: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"available": self.available}
        if self.available:
            out["value"] = self.value
            if self.warmup_sufficient is not None:
                out["warmup_sufficient"] = self.warmup_sufficient
                if not self.warmup_sufficient:
                    out["caution"] = (
                        f"Computed from {self.bars_used} bars; "
                        f"{self.bars_required} recommended for a fully converged "
                        f"value. Treat as provisional."
                    )
        else:
            out["value"] = None
            out["reason"] = self.reason
            if self.bars_used is not None:
                out["bars_available"] = self.bars_used
            if self.bars_required is not None:
                out["bars_required"] = self.bars_required
        return out


def _insufficient(have: int, need: int, what: str) -> IndicatorResult:
    return IndicatorResult(
        available=False,
        reason=(
            f"Insufficient history for {what}: {have} bars available, "
            f"{need} required. Not computed."
        ),
        bars_used=have,
        bars_required=need,
    )


def _round(x: float | None, sig: int = 10) -> float | None:
    """
    Round to `sig` SIGNIFICANT figures, not to a fixed number of decimal
    places.

    This matters for crypto specifically. Fixed decimal rounding (e.g. 6dp)
    is fine for an asset priced at 100000.123456 but silently destroys a token
    priced at 0.00000123456 -- it would collapse to 0.000001, discarding most
    of the value. Since the permitted universe is not restricted to majors,
    prices can span many orders of magnitude, so precision is kept relative to
    the magnitude of the number rather than absolute.
    """
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    fx = float(x)
    if not np.isfinite(fx):
        return None
    if fx == 0.0:
        return 0.0
    return float(f"{fx:.{sig}g}")


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int) -> IndicatorResult:
    """Simple moving average of the most recent `period` closes."""
    n = len(close)
    if n < period:
        return _insufficient(n, period, f"SMA({period})")
    val = close.rolling(window=period).mean().iloc[-1]
    if pd.isna(val):
        return _insufficient(n, period, f"SMA({period})")
    return IndicatorResult(value=_round(val), available=True, bars_used=n)


def ema(close: pd.Series, period: int) -> IndicatorResult:
    """
    Exponential moving average, recursive form (adjust=False), which is what
    charting platforms display: EMA_t = alpha*P_t + (1-alpha)*EMA_{t-1},
    alpha = 2/(period+1), seeded with the first observation.
    """
    n = len(close)
    if n < period:
        return _insufficient(n, period, f"EMA({period})")
    series = close.ewm(span=period, adjust=False).mean()
    val = series.iloc[-1]
    if pd.isna(val):
        return _insufficient(n, period, f"EMA({period})")
    warm_need = _ema_warmup(period)
    return IndicatorResult(
        value=_round(val),
        available=True,
        warmup_sufficient=n >= warm_need,
        bars_used=n,
        bars_required=warm_need,
    )


# ---------------------------------------------------------------------------
# RSI (Wilder)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> IndicatorResult:
    """
    Wilder's Relative Strength Index.

    Gains and losses are smoothed with Wilder's moving average, which is an
    EWMA with alpha = 1/period (equivalently span = 2*period-1).

    Edge cases handled explicitly:
      - all gains, no losses  -> RSI = 100
      - all losses, no gains  -> RSI = 0
      - no movement at all    -> RSI = 50 by convention (0/0 is undefined)
    """
    n = len(close)
    need = period + 1  # need `period` deltas
    if n < need:
        return _insufficient(n, need, f"RSI({period})")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    g = float(avg_gain.iloc[-1])
    l = float(avg_loss.iloc[-1])

    if l == 0.0 and g == 0.0:
        val = 50.0
    elif l == 0.0:
        val = 100.0
    elif g == 0.0:
        val = 0.0
    else:
        rs = g / l
        val = 100.0 - (100.0 / (1.0 + rs))

    warm_need = _wilder_warmup(period)
    return IndicatorResult(
        value=_round(val),
        available=True,
        warmup_sufficient=n >= warm_need,
        bars_used=n,
        bars_required=warm_need,
    )


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> IndicatorResult:
    """
    MACD line = EMA(fast) - EMA(slow); signal = EMA(MACD, signal);
    histogram = MACD - signal. Also reports whether the MACD line is above the
    signal line, and whether a crossover occurred on the most recent bar.
    """
    n = len(close)
    need = slow + signal
    if n < need:
        return _insufficient(n, need, f"MACD({fast},{slow},{signal})")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return _insufficient(n, need, f"MACD({fast},{slow},{signal})")

    above_now = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    crossed = None
    if len(macd_line) >= 2 and not (
        pd.isna(macd_line.iloc[-2]) or pd.isna(signal_line.iloc[-2])
    ):
        above_prev = bool(macd_line.iloc[-2] > signal_line.iloc[-2])
        if above_now != above_prev:
            crossed = "bullish" if above_now else "bearish"

    # MACD needs the slow EMA to converge, then its signal EMA on top.
    warm_need = _ema_warmup(slow) + _ema_warmup(signal)
    return IndicatorResult(
        value={
            "macd": _round(macd_line.iloc[-1]),
            "signal": _round(signal_line.iloc[-1]),
            "histogram": _round(hist.iloc[-1]),
            "macd_above_signal": above_now,
            "crossover_this_bar": crossed,
        },
        available=True,
        warmup_sufficient=n >= warm_need,
        bars_used=n,
        bars_required=warm_need,
    )


# ---------------------------------------------------------------------------
# ATR (Wilder)
# ---------------------------------------------------------------------------

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> IndicatorResult:
    """
    Average True Range, Wilder-smoothed.

    True Range = max(high-low, |high - prev_close|, |low - prev_close|)

    Also reports ATR as a percentage of the latest close, which is the form
    actually useful for volatility-scaled position sizing and stop placement.
    """
    n = len(close)
    need = period + 1
    if n < need:
        return _insufficient(n, need, f"ATR({period})")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]  # no prior close for the first bar

    atr_series = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    val = float(atr_series.iloc[-1])
    last_close = float(close.iloc[-1])
    pct = (val / last_close * 100.0) if last_close else None

    warm_need = _wilder_warmup(period)
    return IndicatorResult(
        value={
            "atr": _round(val),
            "atr_percent_of_price": _round(pct),
        },
        available=True,
        warmup_sufficient=n >= warm_need,
        bars_used=n,
        bars_required=warm_need,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> IndicatorResult:
    """
    Bollinger Bands: a simple moving average with upper/lower bands set
    `num_std` standard deviations away, computed on the same rolling window.

    Reports the three band values, %B (where price sits between the bands --
    0 at the lower band, 1 at the upper band, and possible to fall outside
    [0, 1] on a genuine breakout), bandwidth as a percent of the middle band,
    and a volatility "squeeze" flag: whether the current bandwidth is
    unusually narrow relative to its own last 40 bars, which often precedes a
    breakout in either direction (not a directional signal by itself).
    Squeeze is reported as None -- not guessed -- when there isn't enough
    history to judge "unusual" against.
    """
    n = len(close)
    if n < period:
        return _insufficient(n, period, f"Bollinger Bands({period})")

    mid_series = close.rolling(window=period).mean()
    std_series = close.rolling(window=period).std(ddof=0)
    if pd.isna(mid_series.iloc[-1]) or pd.isna(std_series.iloc[-1]):
        return _insufficient(n, period, f"Bollinger Bands({period})")

    upper_series = mid_series + num_std * std_series
    lower_series = mid_series - num_std * std_series

    mid_val = float(mid_series.iloc[-1])
    upper_val = float(upper_series.iloc[-1])
    lower_val = float(lower_series.iloc[-1])
    last = float(close.iloc[-1])

    band_width = upper_val - lower_val
    percent_b = (last - lower_val) / band_width if band_width > 0 else None
    bandwidth_pct = (band_width / mid_val * 100.0) if mid_val else None

    squeeze: bool | None = None
    baseline_window = 40
    if bandwidth_pct is not None and n >= period + baseline_window:
        bw_series = (
            (upper_series - lower_series) / mid_series * 100.0
        ).iloc[-baseline_window:].dropna()
        if len(bw_series) >= 20:
            squeeze = bool(bandwidth_pct <= bw_series.quantile(0.10))

    if last > upper_val:
        position = "above upper band (overextended)"
    elif last < lower_val:
        position = "below lower band (overextended)"
    elif percent_b is not None and percent_b >= 0.8:
        position = "near upper band"
    elif percent_b is not None and percent_b <= 0.2:
        position = "near lower band"
    else:
        position = "within bands, no extreme"

    return IndicatorResult(
        value={
            "upper_band": _round(upper_val),
            "middle_band": _round(mid_val),
            "lower_band": _round(lower_val),
            "percent_b": _round(percent_b),
            "bandwidth_percent": _round(bandwidth_pct),
            "position": position,
            "squeeze": squeeze,
            "squeeze_note": (
                "True means current bandwidth is in the narrowest 10% of the "
                "last 40 bars -- a volatility contraction that often precedes "
                "a breakout in either direction, not a directional signal by "
                "itself. None means not enough history to judge."
            ),
        },
        available=True,
        bars_used=n,
    )


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> IndicatorResult:
    """
    Stochastic RSI: the Stochastic oscillator formula applied to RSI values
    instead of price. Oscillates strictly between 0 and 100 and reacts faster
    than plain RSI, at the cost of more noise -- useful for timing
    entries/exits on the shorter timeframe under review, not for establishing
    trend by itself.

    %K = SMA(k_smooth) of the raw stochastic of RSI over `stoch_period` bars.
    %D = SMA(d_smooth) of %K.
    """
    n = len(close)
    need = rsi_period + stoch_period + k_smooth + d_smooth
    if n < need:
        return _insufficient(n, need, f"Stochastic RSI({rsi_period},{stoch_period})")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

    g = avg_gain.to_numpy()
    l = avg_loss.to_numpy()
    safe_l = np.where(l == 0.0, 1.0, l)  # placeholder only; unused where l==0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = g / safe_l
        raw_rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_vals = np.where(
        (g == 0.0) & (l == 0.0), 50.0,
        np.where(l == 0.0, 100.0, np.where(g == 0.0, 0.0, raw_rsi)),
    )
    rsi_series = pd.Series(rsi_vals, index=close.index)

    min_rsi = rsi_series.rolling(window=stoch_period).min()
    max_rsi = rsi_series.rolling(window=stoch_period).max()
    denom = max_rsi - min_rsi
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_k = (rsi_series - min_rsi) / denom * 100.0
    # A perfectly flat RSI over the window makes the stochastic undefined;
    # convention treats it as mid-range rather than leaving it NaN.
    raw_k = raw_k.where(denom > 0.0, 50.0)

    k = raw_k.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()

    if pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return _insufficient(n, need, f"Stochastic RSI({rsi_period},{stoch_period})")

    k_val = float(k.iloc[-1])
    d_val = float(d.iloc[-1])

    if k_val >= 80.0:
        zone = "overbought"
    elif k_val <= 20.0:
        zone = "oversold"
    else:
        zone = "neutral"

    crossed = None
    if len(k) >= 2 and not (pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2])):
        above_now = k_val > d_val
        above_prev = bool(k.iloc[-2] > d.iloc[-2])
        if above_now != above_prev:
            crossed = "bullish" if above_now else "bearish"

    warm_need = _wilder_warmup(rsi_period) + stoch_period + k_smooth + d_smooth
    return IndicatorResult(
        value={
            "k": _round(k_val),
            "d": _round(d_val),
            "zone": zone,
            "crossover_this_bar": crossed,
        },
        available=True,
        warmup_sufficient=n >= warm_need,
        bars_used=n,
        bars_required=warm_need,
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def volume_profile(volume: pd.Series, period: int = 20) -> IndicatorResult:
    """Latest bar volume relative to its trailing average."""
    n = len(volume)
    if n < period:
        return _insufficient(n, period, f"volume average({period})")
    avg = float(volume.rolling(window=period).mean().iloc[-1])
    latest = float(volume.iloc[-1])
    if avg <= 0:
        return IndicatorResult(
            available=False,
            reason="Trailing average volume is zero; ratio undefined.",
            bars_used=n,
        )
    ratio = latest / avg
    if ratio >= 2.0:
        desc = "very high (>=2x average)"
    elif ratio >= 1.3:
        desc = "above average"
    elif ratio >= 0.7:
        desc = "near average"
    elif ratio >= 0.4:
        desc = "below average"
    else:
        desc = "very low (<0.4x average)"
    return IndicatorResult(
        value={
            "latest_volume": _round(latest),
            "average_volume": _round(avg),
            "ratio_to_average": _round(ratio),
            "description": desc,
        },
        available=True,
        bars_used=n,
    )


# ---------------------------------------------------------------------------
# Swing points, support / resistance
# ---------------------------------------------------------------------------

def _swing_points(
    high: pd.Series, low: pd.Series, left: int, right: int
) -> tuple[list[dict], list[dict]]:
    """
    Fractal swing detection. A swing high is a bar whose high is strictly
    greater than the `left` bars before and `right` bars after it. Bars within
    `right` of the end are excluded because their status is not yet confirmed
    -- this is deliberate: an unconfirmed swing is not a swing.
    """
    highs, lows = [], []
    n = len(high)
    for i in range(left, n - right):
        window_h = high.iloc[i - left : i + right + 1]
        if high.iloc[i] == window_h.max() and (window_h == high.iloc[i]).sum() == 1:
            highs.append({"index": i, "price": float(high.iloc[i])})
        window_l = low.iloc[i - left : i + right + 1]
        if low.iloc[i] == window_l.min() and (window_l == low.iloc[i]).sum() == 1:
            lows.append({"index": i, "price": float(low.iloc[i])})
    return highs, lows


def swing_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left: int = 3,
    right: int = 3,
    max_points: int = 5,
) -> IndicatorResult:
    """Recent confirmed swing highs/lows and the market structure they imply."""
    n = len(high)
    need = left + right + 1
    if n < need:
        return _insufficient(n, need, "swing structure")

    highs, lows = _swing_points(high, low, left, right)
    if not highs and not lows:
        return IndicatorResult(
            available=False,
            reason="No confirmed swing points found in the available history.",
            bars_used=n,
        )

    recent_highs = highs[-max_points:]
    recent_lows = lows[-max_points:]

    structure = "indeterminate"
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        hh = recent_highs[-1]["price"] > recent_highs[-2]["price"]
        hl = recent_lows[-1]["price"] > recent_lows[-2]["price"]
        lh = recent_highs[-1]["price"] < recent_highs[-2]["price"]
        ll = recent_lows[-1]["price"] < recent_lows[-2]["price"]
        if hh and hl:
            structure = "uptrend (higher highs, higher lows)"
        elif lh and ll:
            structure = "downtrend (lower highs, lower lows)"
        else:
            structure = "mixed / ranging"

    last = float(close.iloc[-1])
    res = [h["price"] for h in highs if h["price"] > last]
    sup = [l["price"] for l in lows if l["price"] < last]

    return IndicatorResult(
        value={
            "recent_swing_highs": [_round(h["price"]) for h in recent_highs],
            "recent_swing_lows": [_round(l["price"]) for l in recent_lows],
            "market_structure": structure,
            "nearest_resistance": _round(min(res)) if res else None,
            "nearest_support": _round(max(sup)) if sup else None,
            "note": (
                "Swing points within the last "
                f"{right} bars are excluded as unconfirmed."
            ),
        },
        available=True,
        bars_used=n,
    )


# ---------------------------------------------------------------------------
# Fibonacci retracement
# ---------------------------------------------------------------------------

def fibonacci_levels(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 90
) -> IndicatorResult:
    """
    Retracement levels across the highest high and lowest low of the lookback
    window, oriented by which came first (so an up-leg and a down-leg produce
    correctly ordered levels).
    """
    n = len(high)
    if n < 10:
        return _insufficient(n, 10, "Fibonacci retracement")

    window = min(lookback, n)
    h_win = high.iloc[-window:]
    l_win = low.iloc[-window:]

    hi = float(h_win.max())
    lo = float(l_win.min())
    if hi <= lo:
        return IndicatorResult(
            available=False,
            reason="Degenerate price range in lookback window.",
            bars_used=n,
        )

    hi_pos = int(h_win.values.argmax())
    lo_pos = int(l_win.values.argmin())
    direction = "up_leg" if lo_pos < hi_pos else "down_leg"

    rng = hi - lo
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    if direction == "up_leg":
        # retracing down from the high
        levels = {f"{r:.3f}": _round(hi - rng * r) for r in ratios}
    else:
        # retracing up from the low
        levels = {f"{r:.3f}": _round(lo + rng * r) for r in ratios}

    return IndicatorResult(
        value={
            "swing_high": _round(hi),
            "swing_low": _round(lo),
            "leg_direction": direction,
            "levels": levels,
            "current_price": _round(float(close.iloc[-1])),
            "lookback_bars": window,
        },
        available=True,
        bars_used=n,
    )


# ---------------------------------------------------------------------------
# Trend summary
# ---------------------------------------------------------------------------

def trend_summary(close: pd.Series) -> IndicatorResult:
    """
    Price position relative to the 20/50/200 SMAs, plus MA stacking.
    Reports only the MAs that genuinely have enough history.
    """
    n = len(close)
    last = float(close.iloc[-1])
    out: dict[str, Any] = {"current_price": _round(last)}

    mas: dict[int, float | None] = {}
    for p in (20, 50, 200):
        r = sma(close, p)
        if r.available:
            mas[p] = float(r.value)
            out[f"sma_{p}"] = r.value
            out[f"price_vs_sma_{p}"] = (
                "above" if last > r.value else "below" if last < r.value else "at"
            )
        else:
            mas[p] = None
            out[f"sma_{p}"] = None
            out[f"price_vs_sma_{p}"] = f"unavailable ({n} bars, {p} required)"

    if all(mas[p] is not None for p in (20, 50, 200)):
        if mas[20] > mas[50] > mas[200]:
            out["ma_alignment"] = "bullish (20 > 50 > 200)"
        elif mas[20] < mas[50] < mas[200]:
            out["ma_alignment"] = "bearish (20 < 50 < 200)"
        else:
            out["ma_alignment"] = "mixed"
    else:
        out["ma_alignment"] = "unavailable (requires 200 bars)"

    for label, periods in (("change_1_bar", 1), ("change_7_bar", 7), ("change_30_bar", 30)):
        if n > periods:
            prior = float(close.iloc[-1 - periods])
            out[label + "_percent"] = _round((last / prior - 1.0) * 100.0) if prior else None
        else:
            out[label + "_percent"] = None

    return IndicatorResult(value=out, available=True, bars_used=n)


# ---------------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------------

def relative_strength(
    asset_close: pd.Series, benchmark_close: pd.Series, periods: int = 30
) -> IndicatorResult:
    """
    Percentage performance of the asset against a benchmark over `periods`
    bars. Both series must have enough history; no interpolation is performed.
    """
    na, nb = len(asset_close), len(benchmark_close)
    need = periods + 1
    if na < need or nb < need:
        return _insufficient(min(na, nb), need, f"relative strength({periods})")

    a0, a1 = float(asset_close.iloc[-need]), float(asset_close.iloc[-1])
    b0, b1 = float(benchmark_close.iloc[-need]), float(benchmark_close.iloc[-1])
    if a0 <= 0 or b0 <= 0:
        return IndicatorResult(
            available=False, reason="Non-positive starting price.", bars_used=min(na, nb)
        )

    a_ret = (a1 / a0 - 1.0) * 100.0
    b_ret = (b1 / b0 - 1.0) * 100.0
    spread = a_ret - b_ret

    return IndicatorResult(
        value={
            "asset_return_percent": _round(a_ret),
            "benchmark_return_percent": _round(b_ret),
            "outperformance_percent": _round(spread),
            "verdict": (
                "outperforming" if spread > 0.5
                else "underperforming" if spread < -0.5
                else "in line"
            ),
            "periods": periods,
        },
        available=True,
        bars_used=min(na, nb),
    )


# ---------------------------------------------------------------------------
# Multi-timeframe confluence
# ---------------------------------------------------------------------------

def _timeframe_bias(bundle: dict[str, Any]) -> dict[str, Any]:
    """
    A simple, auditable directional score for one timeframe's indicator
    bundle, used only to build cross-timeframe confluence -- it is not a
    trade signal by itself. Each available signal below contributes +1
    (bullish) or -1 (bearish); a signal that came back unavailable
    contributes nothing rather than being guessed.
    """
    score = 0
    signals: list[str] = []

    trend = bundle.get("trend", {})
    if trend.get("available"):
        alignment = trend.get("value", {}).get("ma_alignment", "")
        if alignment.startswith("bullish"):
            score += 1
            signals.append("MA alignment bullish")
        elif alignment.startswith("bearish"):
            score -= 1
            signals.append("MA alignment bearish")

    swing = bundle.get("swing_structure", {})
    if swing.get("available"):
        structure = swing.get("value", {}).get("market_structure", "")
        if structure.startswith("uptrend"):
            score += 1
            signals.append("swing structure uptrend")
        elif structure.startswith("downtrend"):
            score -= 1
            signals.append("swing structure downtrend")

    macd_entry = bundle.get("macd", {})
    if macd_entry.get("available"):
        if macd_entry.get("value", {}).get("macd_above_signal"):
            score += 1
            signals.append("MACD above signal")
        else:
            score -= 1
            signals.append("MACD below signal")

    rsi_entry = bundle.get("rsi_14", {})
    if rsi_entry.get("available"):
        val = rsi_entry.get("value")
        if isinstance(val, (int, float)):
            if val > 50.0:
                score += 1
                signals.append("RSI > 50")
            elif val < 50.0:
                score -= 1
                signals.append("RSI < 50")

    stoch_entry = bundle.get("stochastic_rsi", {})
    if stoch_entry.get("available"):
        crossed = stoch_entry.get("value", {}).get("crossover_this_bar")
        if crossed == "bullish":
            score += 1
            signals.append("Stochastic RSI bullish crossover")
        elif crossed == "bearish":
            score -= 1
            signals.append("Stochastic RSI bearish crossover")

    bias = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    return {"bias": bias, "score": score, "signals_used": signals}


def multi_timeframe_confluence(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Cross-timeframe confluence summary from per-timeframe indicator bundles
    (as produced by compute_all), keyed by timeframe label (e.g. "1d", "4h",
    "1h").

    For each timeframe, a directional bias is derived only from whichever
    signals actually came back available (MA alignment, swing structure,
    MACD, RSI, Stochastic RSI crossover) -- never guessed for a signal
    reported unavailable. Timeframes are then compared: how many agree on
    direction, and whether that agreement is unanimous.

    This does not decide whether to trade. It is one more piece of evidence
    for the decision hierarchy in the strategy rules ("weight decisions by
    how many independent forms of evidence agree"), to be weighed alongside
    everything else -- not a standalone signal.
    """
    per_tf = {tf: _timeframe_bias(bundle) for tf, bundle in bundles.items()}
    bullish = [tf for tf, r in per_tf.items() if r["bias"] == "bullish"]
    bearish = [tf for tf, r in per_tf.items() if r["bias"] == "bearish"]
    neutral = [tf for tf, r in per_tf.items() if r["bias"] == "neutral"]

    total = len(per_tf)
    non_neutral = len(bullish) + len(bearish)
    aligned = non_neutral >= 2 and (len(bullish) == non_neutral or len(bearish) == non_neutral)

    if bullish and not bearish:
        overall = (
            f"bullish confluence ({len(bullish)}/{total} timeframes bullish, "
            f"{len(neutral)} neutral)"
        )
    elif bearish and not bullish:
        overall = (
            f"bearish confluence ({len(bearish)}/{total} timeframes bearish, "
            f"{len(neutral)} neutral)"
        )
    elif bullish and bearish:
        overall = (
            f"conflicting ({len(bullish)} bullish vs {len(bearish)} bearish, "
            f"{len(neutral)} neutral) -- signals disagree across timeframes"
        )
    else:
        overall = f"no directional lean ({len(neutral)}/{total} timeframes neutral)"

    total_for_score = max(total, 1)
    confluence_score = round((len(bullish) - len(bearish)) / total_for_score, 3)

    return {
        "per_timeframe": per_tf,
        "timeframes_bullish": len(bullish),
        "timeframes_bearish": len(bearish),
        "timeframes_neutral": len(neutral),
        "aligned": aligned,
        "overall": overall,
        "confluence_score": confluence_score,
        "confluence_score_note": (
            "Numeric form of the same verdict, from -1.0 (all timeframes "
            "bearish) to +1.0 (all bullish), 0.0 for no lean or an even split. "
            "Useful for mechanically scaling position size (e.g. half the "
            "normal risk budget when abs(confluence_score) < 0.6) rather than "
            "branching on the text label."
        ),
        "note": (
            "Each timeframe's bias is a simple count of available directional "
            "signals (MA alignment, swing structure, MACD, RSI, Stochastic "
            "RSI crossover); unavailable signals are excluded, never guessed. "
            "This is supporting evidence for the decision hierarchy, not a "
            "trade signal by itself."
        ),
    }


# ---------------------------------------------------------------------------
# Cross-asset correlation
# ---------------------------------------------------------------------------

def correlation_matrix(
    closes: dict[str, pd.Series], lookback: int = 30
) -> dict[str, Any]:
    """
    Pearson correlation of daily returns between every pair of assets in
    `closes` (symbol -> close-price Series, oldest first), over the trailing
    `lookback` daily bars, plus each asset's correlation to whichever series
    is keyed "BTC" if present.

    Used to approximate a portfolio's real diversification: several
    high-beta altcoins held at once can be functionally one leveraged BTC
    bet even when each individually respects a per-position size cap, if
    they are all highly correlated to BTC (and to each other).

    An asset with fewer than `lookback + 1` closes is excluded from the
    matrix entirely rather than computed from a shorter, silently
    mismatched window.
    """
    returns: dict[str, pd.Series] = {}
    excluded: list[str] = []
    for sym, series in closes.items():
        s = series.dropna()
        if len(s) < lookback + 1:
            excluded.append(sym)
            continue
        r = s.pct_change().dropna().iloc[-lookback:]
        if len(r) < lookback:
            excluded.append(sym)
            continue
        returns[sym] = r.reset_index(drop=True)

    symbols = sorted(returns.keys())
    if len(symbols) < 2:
        return {
            "available": False,
            "reason": (
                f"Fewer than two symbols had {lookback}+ days of aligned "
                f"return history; excluded: {excluded}"
            ),
            "excluded_insufficient_history": excluded,
        }

    df = pd.DataFrame({s: returns[s] for s in symbols})
    corr = df.corr(method="pearson")

    pairs: dict[str, float] = {}
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            val = corr.loc[a, b]
            pairs[f"{a}:{b}"] = _round(float(val)) if pd.notna(val) else None

    to_btc: dict[str, float] = {}
    if "BTC" in symbols:
        for s in symbols:
            if s == "BTC":
                continue
            val = corr.loc[s, "BTC"]
            to_btc[s] = _round(float(val)) if pd.notna(val) else None

    return {
        "available": True,
        "lookback_days": lookback,
        "pairs": pairs,
        "to_btc": to_btc,
        "excluded_insufficient_history": excluded,
        "note": (
            "Pearson correlation of daily percent returns. Above ~0.75 to BTC "
            "is treated as high-correlation by the strategy's portfolio "
            "diversification rule -- holding more than two such positions at "
            "once is functionally one leveraged BTC bet even if each "
            "individually respects per-position size limits."
        ),
    }


# ---------------------------------------------------------------------------
# Trade validation -- moves reward-to-risk and sizing arithmetic out of prose
# ---------------------------------------------------------------------------

def suggest_position_size(
    account_equity_usd: float,
    entry_price: float,
    collar_adjusted_stop_price: float,
    risk_budget_pct: float = 2.5,
    min_position_usd: float = 50.0,
    max_position_pct: float = 40.0,
) -> dict[str, Any]:
    """
    Derive position size from a risk budget, rather than a chosen or default
    dollar amount.

    size = (risk_budget_pct% of account equity) / (collar-adjusted stop
    distance, as a percent of entry price)

    bounded below by `min_position_usd` and above by `max_position_pct`% of
    equity -- but critically, if the risk-budget-derived size comes out BELOW
    the floor, that means the stop is too wide for this account to risk at
    the standing budget. This function reports that as a rejection
    (binding_constraint = "min_floor_unmet"), not as a silently-rounded-up
    minimum position, because rounding up would mean accepting more risk than
    the budget allows just to meet the floor.
    """
    if entry_price <= 0 or account_equity_usd <= 0:
        return {
            "available": False,
            "reason": "entry_price and account_equity_usd must both be positive.",
        }
    if collar_adjusted_stop_price >= entry_price:
        return {
            "available": False,
            "reason": (
                "collar_adjusted_stop_price must be below entry_price for a "
                "long position -- a stop at or above entry is not a valid "
                "invalidation level."
            ),
        }

    stop_distance_pct = (entry_price - collar_adjusted_stop_price) / entry_price
    risk_usd_budget = (risk_budget_pct / 100.0) * account_equity_usd
    raw_size_usd = risk_usd_budget / stop_distance_pct
    max_usd = (max_position_pct / 100.0) * account_equity_usd

    if raw_size_usd < min_position_usd:
        return {
            "available": True,
            "size_usd": None,
            "size_units": None,
            "risk_usd": None,
            "binding_constraint": "min_floor_unmet",
            "raw_risk_budget_size_usd": _round(raw_size_usd),
            "stop_distance_percent": _round(stop_distance_pct * 100.0),
            "verdict": "reject",
            "reason": (
                f"At a {risk_budget_pct}% risk budget, this stop distance "
                f"({_round(stop_distance_pct * 100.0)}%) implies a position of "
                f"only ${_round(raw_size_usd)}, below the ${min_position_usd:g} "
                f"floor. This means the stop is too wide for what this account "
                f"can safely risk on it -- do not round up to the floor; the "
                f"setup does not qualify at this account size."
            ),
        }

    if raw_size_usd > max_usd:
        size_usd = max_usd
        binding = "max_position_pct_cap"
    else:
        size_usd = raw_size_usd
        binding = "risk_budget"

    return {
        "available": True,
        "size_usd": _round(size_usd),
        "size_units": _round(size_usd / entry_price),
        "risk_usd": _round(size_usd * stop_distance_pct),
        "binding_constraint": binding,
        "stop_distance_percent": _round(stop_distance_pct * 100.0),
        "verdict": "pass",
    }


def validate_trade_setup(
    entry_price: float,
    stop_trigger_price: float,
    target_price: float,
    account_equity_usd: float,
    current_open_risk_usd: float = 0.0,
    round_trip_cost_pct: float = 0.0,
    risk_budget_pct: float = 2.5,
    reward_to_risk_floor: float = 1.5,
    reward_to_risk_exception_floor: float = 1.2,
    allow_exception: bool = False,
    min_position_usd: float = 50.0,
    max_position_pct: float = 40.0,
    portfolio_risk_cap_pct: float = 8.0,
) -> dict[str, Any]:
    """
    A single, hard, external gate that a proposed long spot entry must clear
    before an order is placed -- this replaces doing the reward-to-risk and
    sizing arithmetic in prose, which is exactly how a past run inverted a
    reward-to-risk calculation (stating ~0.3-0.6:1 for a setup whose own cited
    numbers actually worked out to ~1.36-2.73:1). The math happens here once,
    the same way every time.

    Risk is computed off a COLLAR-ADJUSTED stop, not the raw stop_trigger_price:
    a live stop_loss order's actual fill collar sits roughly 5% below its
    trigger for a sell stop, so collar_adjusted_stop = stop_trigger_price *
    0.95. A ratio computed off the raw trigger flatters itself by roughly 25%.

    round_trip_cost_pct (from preview_crypto_order's fee estimate, as a
    percent of notional, e.g. 0.5 for 0.5%) is subtracted from the reward leg
    before the ratio is computed, so trading costs are not merely a
    qualitative afterthought.

    Set allow_exception=True only when the calling thesis has already named a
    separate, specific, concrete reason the conventional target understates
    the opportunity -- this function does not and cannot judge that itself;
    it only lets a ratio between reward_to_risk_exception_floor and
    reward_to_risk_floor pass when explicitly told to. The calling agent is
    responsible for tracking that this exception is used at most once per ten
    trades and for flagging it in the journal and summary -- this function
    has no memory across calls.

    Args:
        entry_price: Proposed fill price.
        stop_trigger_price: Proposed protective stop's trigger price (not
            the collar-adjusted figure -- that is computed here).
        target_price: Nearest technically defensible target.
        account_equity_usd: Current total account value.
        current_open_risk_usd: Sum of collar-adjusted dollar risk already
            committed across all other open positions, before this trade.
        round_trip_cost_pct: Estimated round-trip trading cost as a percent
            of notional, from preview_crypto_order.
        risk_budget_pct: Percent of equity to risk on this trade (2.5 under
            normal conditions; lower this directly when the drawdown-based
            risk controls call for it).
        reward_to_risk_floor: Hard floor, normally 1.5.
        reward_to_risk_exception_floor: Absolute floor even with the
            documented exception, normally 1.2.
        allow_exception: Whether the documented, bounded exception applies
            to this specific trade (see above).
        min_position_usd: Position size floor, normally 50.
        max_position_pct: Position size ceiling as percent of equity,
            normally 40.
        portfolio_risk_cap_pct: Maximum total collar-adjusted open risk
            across the whole portfolio after this trade, normally 8.
    """
    reasons: list[str] = []

    if entry_price <= 0 or account_equity_usd <= 0:
        return {
            "verdict": "fail",
            "reasons": ["invalid_inputs"],
            "detail": "entry_price and account_equity_usd must both be positive.",
        }

    collar_adjusted_stop = stop_trigger_price * 0.95
    if collar_adjusted_stop >= entry_price:
        return {
            "verdict": "fail",
            "reasons": ["invalid_stop"],
            "collar_adjusted_stop": _round(collar_adjusted_stop),
            "detail": (
                "Collar-adjusted stop is at or above entry price -- not a "
                "valid invalidation level for a long position."
            ),
        }

    risk_per_unit = entry_price - collar_adjusted_stop
    risk_pct = risk_per_unit / entry_price
    risk_usd_amount = risk_per_unit  # per unit; scaled by size below

    reward_per_unit_gross = target_price - entry_price
    reward_pct_gross = reward_per_unit_gross / entry_price
    reward_pct_net = reward_pct_gross - (round_trip_cost_pct / 100.0)

    if reward_pct_net <= 0:
        reward_to_risk = 0.0
        reasons.append("rr_below_floor")
    else:
        reward_to_risk = reward_pct_net / risk_pct

        if reward_to_risk < reward_to_risk_exception_floor:
            reasons.append("rr_below_floor")
        elif reward_to_risk < reward_to_risk_floor:
            if not allow_exception:
                reasons.append("rr_below_floor")
            # else: in the narrow 1.2-1.5 band with the exception explicitly
            # invoked by the caller -- no reason appended, but flagged below.

    sizing = suggest_position_size(
        account_equity_usd=account_equity_usd,
        entry_price=entry_price,
        collar_adjusted_stop_price=collar_adjusted_stop,
        risk_budget_pct=risk_budget_pct,
        min_position_usd=min_position_usd,
        max_position_pct=max_position_pct,
    )

    position_size_usd = None
    risk_usd = None
    portfolio_risk_pct_after = None

    if not sizing.get("available"):
        reasons.append("sizing_error")
    elif sizing.get("verdict") == "reject":
        reasons.append("size_below_50_floor")
        reasons.append("stop_distance_exceeds_max")
    else:
        position_size_usd = sizing["size_usd"]
        risk_usd = sizing["risk_usd"]
        portfolio_risk_pct_after = (
            (current_open_risk_usd + risk_usd) / account_equity_usd * 100.0
        )
        if portfolio_risk_pct_after > portfolio_risk_cap_pct:
            reasons.append("portfolio_risk_cap_exceeded")

    verdict = "pass" if not reasons else "fail"

    return {
        "collar_adjusted_stop": _round(collar_adjusted_stop),
        "risk_pct": _round(risk_pct * 100.0),
        "risk_usd_per_unit": _round(risk_usd_amount),
        "reward_pct_net_of_costs": _round(reward_pct_net * 100.0),
        "reward_usd_per_unit": _round(reward_per_unit_gross),
        "reward_to_risk": _round(reward_to_risk),
        "reward_to_risk_exception_used": (
            allow_exception
            and reward_to_risk_exception_floor <= reward_to_risk < reward_to_risk_floor
        ),
        "position_size_usd": position_size_usd,
        "position_size_detail": sizing,
        "portfolio_risk_pct_after": (
            _round(portfolio_risk_pct_after) if portfolio_risk_pct_after is not None else None
        ),
        "verdict": verdict,
        "reasons": reasons,
        "instruction_to_agent": (
            "No order may be placed unless verdict is 'pass'. Quote this "
            "entire result verbatim in the trade thesis -- it is the "
            "reward-to-risk and sizing arithmetic, computed once, not "
            "something to redo or round in prose."
        ),
    }


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute the full indicator set from an OHLCV frame.

    `df` must have columns: open, high, low, close, volume -- oldest row first.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")

    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])

    return {
        "bars_available": len(df),
        "trend": trend_summary(c).to_dict(),
        "ema_20": ema(c, 20).to_dict(),
        "ema_50": ema(c, 50).to_dict(),
        "ema_200": ema(c, 200).to_dict(),
        "rsi_14": rsi(c, 14).to_dict(),
        "stochastic_rsi": stochastic_rsi(c).to_dict(),
        "macd": macd(c).to_dict(),
        "bollinger_bands": bollinger_bands(c).to_dict(),
        "atr_14": atr(h, l, c, 14).to_dict(),
        "volume": volume_profile(v, 20).to_dict(),
        "swing_structure": swing_structure(h, l, c).to_dict(),
        "fibonacci": fibonacci_levels(h, l, c).to_dict(),
    }
