"""
Deterministic tests for crosssection.py.

No network. Synthetic candle frames with known properties are fed in and the
outputs are checked against hand-computed expectations, so a regression shows
up as a failing assertion rather than a plausible-looking number.

Run from the repo root:  python test_crosssection.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import crosssection as cs
import datasource as ds

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def make_frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build an OHLCV frame from a close series, with a plausible bar range."""
    c = np.asarray(closes, dtype=float)
    vol = np.asarray(volumes if volumes is not None else [1000.0] * len(c), dtype=float)
    return pd.DataFrame(
        {
            "open": np.concatenate([[c[0]], c[:-1]]),
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": vol,
        }
    )


def drifting(start: float, daily_pct: float, n: int, seed: int,
             sigma: float = 0.003) -> list[float]:
    """
    A price path with a fixed daily drift plus small seeded noise.

    The noise matters: a perfectly smooth series has zero realized volatility,
    and the volatility-adjusted momentum component correctly refuses to divide
    by it -- which would leave every asset without a composite score. Real
    price series always carry noise, so the test data must too.
    """
    rng = np.random.default_rng(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * (1.0 + daily_pct / 100.0 + rng.normal(0.0, sigma)))
    return out


N = 400


def build_universe() -> dict[str, pd.DataFrame]:
    """
    Twelve assets with deliberately different behaviour.

    BTC is the benchmark. SPIKE tracks BTC until the final bar, then jumps --
    so its rank should rise and accelerate. FADE does the reverse.
    """
    frames = {"BTC": make_frame(drifting(50_000, 0.10, N, seed=1))}
    # A spread of steady out/under-performers so ranking has something to rank.
    for i, rate in enumerate([0.30, 0.25, 0.20, 0.15, 0.05, 0.00, -0.05, -0.15, -0.25]):
        frames[f"A{i}"] = make_frame(drifting(100, rate, N, seed=10 + i))

    spike = drifting(100, 0.10, N, seed=90)
    spike[-1] = spike[-1] * 1.12  # a sharp final-bar move
    frames["SPIKE"] = make_frame(spike)

    fade = drifting(100, 0.10, N, seed=91)
    fade[-1] = fade[-1] * 0.88
    frames["FADE"] = make_frame(fade)
    return frames


print("=== 1. as-of truncation is leak-free ===")
U = build_universe()
K = 3
from_full = cs._component_values(U, K)
truncated = {s: d.iloc[: len(d) - K] for s, d in U.items()}
from_trunc = cs._component_values(truncated, 0)
mismatch = [
    s
    for s in U
    if from_full.get(s, {}).get("returns_percent")
    != from_trunc.get(s, {}).get("returns_percent")
]
check("offset-k on full frame == offset-0 on truncated frame", not mismatch, str(mismatch))

# The final bar must be invisible at offset 1. SPIKE's jump is on that bar.
spike_now = cs._component_values(U, 0)["SPIKE"]["returns_percent"]["1d"]
spike_prev = cs._component_values(U, 1)["SPIKE"]["returns_percent"]["1d"]
check(
    "final-bar move is invisible one bar back",
    spike_now > 10.0 and spike_prev < 2.0,
    f"now={spike_now:.2f}% prev={spike_prev:.2f}%",
)

print("\n=== 2. rank trajectory reflects real movement ===")
st = cs.cross_sectional_strength(U)
A = st["assets"]
check(
    "every asset got a composite score",
    all(A[s].get("composite_percentile") is not None for s in U),
    str({s: A[s].get("composite_percentile") for s in U if A[s].get("composite_percentile") is None}),
)
check(
    "SPIKE rank rises and accelerates",
    A["SPIKE"]["rank_change_1d"] > 0 and A["SPIKE"]["rank_acceleration"] > 0,
    f"chg1d={A['SPIKE']['rank_change_1d']} accel={A['SPIKE']['rank_acceleration']}",
)
check(
    "FADE rank falls and decelerates",
    A["FADE"]["rank_change_1d"] < 0 and A["FADE"]["rank_acceleration"] < 0,
    f"chg1d={A['FADE']['rank_change_1d']} accel={A['FADE']['rank_acceleration']}",
)
steady_moves = [abs(A[f"A{i}"]["rank_change_1d"]) for i in range(9)]
check(
    "steady performers move far less than SPIKE",
    float(np.median(steady_moves)) < abs(A["SPIKE"]["rank_change_1d"]),
    f"median_steady={np.median(steady_moves):.1f} spike={A['SPIKE']['rank_change_1d']:.1f}",
)
check(
    "strongest steady performer outranks the weakest",
    A["A0"]["composite_percentile"] > A["A8"]["composite_percentile"],
    f"A0={A['A0']['composite_percentile']} A8={A['A8']['composite_percentile']}",
)

print("\n=== 3. percentiles and ranks are well formed ===")
pcts = [d["composite_percentile"] for d in A.values() if d.get("composite_percentile") is not None]
ranks = sorted(d["composite_rank"] for d in A.values() if d.get("composite_rank"))
check("percentiles within [0,100]", all(0 <= p <= 100 for p in pcts), f"n={len(pcts)}")
check("top is 100, bottom is 0", abs(max(pcts) - 100) < 1e-6 and abs(min(pcts)) < 1e-6)
check("ranks are dense 1..n", ranks == list(range(1, len(ranks) + 1)), f"n={len(ranks)}")

print("\n=== 4. relative-strength arithmetic ===")
# Recompute every asset's excess return straight from raw closes. Comparing
# stored-minus-stored against stored would fail on rounding alone, since the
# module reports to six significant figures; this compares against the
# unrounded truth instead.
btc_raw = float(U["BTC"]["close"].iloc[-1]) / float(U["BTC"]["close"].iloc[-8]) - 1
bad = []
for s, d in A.items():
    if not d.get("rankable"):
        continue
    c = U[s]["close"]
    expect = (float(c.iloc[-1]) / float(c.iloc[-8]) - 1) * 100 - btc_raw * 100
    got = d["vs_btc_percent"]["7d"]
    if abs(got - expect) > 1e-4 * max(1.0, abs(expect)):
        bad.append((s, got, expect))
check("vs_btc matches a raw-close recomputation for every asset", not bad, str(bad))
check("BTC vs itself is zero", abs(A["BTC"]["vs_btc_percent"]["7d"]) < 1e-9)
check(
    "volatility-adjusted momentum equals 7d return over 30d realized vol",
    abs(
        A["A0"]["vol_adjusted_momentum_7d"]
        - A["A0"]["returns_percent"]["7d"] / A["A0"]["realized_vol_30d_percent"]
    )
    < 1e-4,
)

print("\n=== 5. short history is refused, never estimated ===")
short = {"BTC": U["BTC"], "TINY": U["A0"].iloc[:40]}
comps = cs._component_values(short, 0)
check("under-length asset is not rankable", comps["TINY"]["rankable"] is False)
check("refusal states why", "required for ranking" in comps["TINY"]["reason"])
check("volatility_state refuses short history",
      cs.volatility_state(U["A0"].iloc[:40]).get("available") is False)

print("\n=== 6. volatility compression then expansion ===")
rng = np.random.default_rng(7)
quiet = [100.0]
for _ in range(119):  # normal volatility
    quiet.append(quiet[-1] * (1 + rng.normal(0, 0.02)))
for _ in range(23):  # compressed
    quiet.append(quiet[-1] * (1 + rng.normal(0, 0.0008)))
for _ in range(7):  # expanding
    quiet.append(quiet[-1] * (1 + rng.normal(0, 0.035)))
vols = [1000.0] * 143 + [2000.0] * 7
squeeze = cs.volatility_state(make_frame(quiet, vols))
check("squeeze frame is analysable", squeeze.get("available"))
check("recent compression detected", squeeze.get("was_compressed_within_10_bars"))
check("bandwidth expanding", squeeze.get("bandwidth_expanding"))
check("volume expansion measured", squeeze["volume_5d_vs_20d"] > 1.1,
      str(squeeze["volume_5d_vs_20d"]))
check("transition flagged", squeeze.get("compression_expansion_transition"))

steady = [100.0]
for _ in range(199):
    steady.append(steady[-1] * (1 + rng.normal(0, 0.02)))
calm = cs.volatility_state(make_frame(steady))
check("steady-volatility asset is not flagged",
      calm.get("compression_expansion_transition") is False)
check("percentiles bounded",
      0 <= calm["bollinger_bandwidth_percentile"] <= 100
      and 0 <= calm["atr_percentile"] <= 100)

print("\n=== 7. breadth arithmetic ===")
br = cs.market_breadth(U)
check("breadth available", br.get("available"))
cur = br["current"]
# Nine steady performers: five rise, four fall. Plus BTC up, SPIKE up, FADE down.
manual_above20 = 0
counted = 0
for sym, df in U.items():
    s20 = cs.ind.sma(df["close"], 20)
    if s20.available:
        counted += 1
        manual_above20 += int(float(df["close"].iloc[-1]) > float(s20.value))
check(
    "percent_above_sma20 matches a manual count",
    abs(cur["percent_above_sma20"] - 100.0 * manual_above20 / counted) < 0.01,
    f"tool={cur['percent_above_sma20']} manual={100.0*manual_above20/counted:.4f}",
)
check("sample sizes reported", cur["sample_sizes"]["sma20"] == counted)
check("classification is a known label",
      br["classification"] in {"broadening", "narrowing", "broad", "weak", "stable", "unknown"},
      f"{br['classification']} ({br['classification_rule']})")
check("rotation reported", br["rotation"] is not None, str(br["rotation"]))

print("\n=== 8. promotion: gate, agreement, suppression, cap ===")
out8 = cs.scan_universe.__wrapped__ if hasattr(cs.scan_universe, "__wrapped__") else None
# Flags are computed per asset against a volume bar drawn from the scan.
vol_states = {s: cs.volatility_state(d) for s, d in U.items()}
bar = cs._volume_bar(vol_states)
check("volume bar respects the absolute floor", bar >= cs.VOLUME_ABSOLUTE_FLOOR, f"bar={bar}")

raw = {}
for s in U:
    f, _ = cs._signal_flags(A[s], vol_states[s], bar)
    raw[s] = f
sup = cs._suppressed_categories(raw, len(U))
for cat, info in sup.items():
    check(f"suppressed {cat} fired on >{cs.COMMON_SIGNAL_FRACTION:.0%}",
          info["fraction"] > cs.COMMON_SIGNAL_FRACTION, str(info["fraction"]))

# A category firing on everything must be suppressed; one firing rarely must not.
everywhere = {s: {"volume": True} for s in U}
check("a category firing on the whole universe is suppressed",
      "volume" in cs._suppressed_categories(everywhere, len(U)))
rare = {s: ({"volume": True} if i == 0 else {}) for i, s in enumerate(U)}
check("a rarely-firing category survives",
      "volume" not in cs._suppressed_categories(rare, len(U)))

print("\n=== 9. fetch failures are surfaced, not swallowed ===")
cs._frame_cache.clear()
real_fetch = ds.fetch_ohlcv


def fake_fetch(symbol, timeframe="1d", drop_incomplete=True):
    s = symbol.upper()
    if s == "BROKEN":
        raise ds.DataSourceError("simulated Kraken outage")
    if s in U:
        return ds.OHLCVResult(
            symbol=s, kraken_pair=f"{s}USD", timeframe=timeframe, df=U[s],
            live_price=None, dropped_incomplete=True,
            last_closed_time="2026-09-22T00:00:00+00:00", fetched_at="now",
        )
    raise ds.DataSourceError(f"no pair for {s}")


ds.fetch_ohlcv = fake_fetch
try:
    frames, failures = cs.fetch_universe(["BTC", "A0", "BROKEN"])
    check("broken symbol appears in failures",
          any(f["symbol"] == "BROKEN" for f in failures), str(failures))
    check("failure carries a reason",
          "simulated Kraken outage" in failures[0]["reason"], str(failures))
    check("healthy symbols still returned", "BTC" in frames and "A0" in frames)

    print("\n=== 10. end-to-end scan ===")
    cs._frame_cache.clear()
    out = cs.scan_universe(list(U.keys()))
    check("scan succeeded", "error" not in out, out.get("error", ""))
    check("summary is the default", out["detail"] == "summary")
    check("summary omits the per-asset block", "assets" not in out)
    check("leaderboard covers every fetched asset",
          len(out["leaderboard"]) == out["fetched"])
    check("leaderboard is sorted by percentile",
          [r["percentile"] for r in out["leaderboard"]]
          == sorted([r["percentile"] for r in out["leaderboard"]], reverse=True))
    check("promoted list respects the cap",
          len(out["promoted_candidates"]) <= cs.MAX_PROMOTED,
          f"{len(out['promoted_candidates'])} of max {cs.MAX_PROMOTED}")
    check("every promoted asset met the agreement requirement",
          all(len(p["categories"]) >= cs.REQUIRED_CATEGORIES
              for p in out["promoted_candidates"]))

    # The exact regression the first live scan exposed: weak assets promoted
    # on volatility and volume alone.
    check("no asset below the strength gate is promoted",
          all(p["composite_percentile"] >= cs.MIN_PROMOTION_PERCENTILE
              for p in out["promoted_candidates"]),
          str([(p["symbol"], p["composite_percentile"]) for p in out["promoted_candidates"]]))
    weak = [r for r in out["leaderboard"]
            if r["percentile"] is not None and r["percentile"] < cs.MIN_PROMOTION_PERCENTILE]
    check("weak half of the universe is all blocked",
          all(r["promoted"] is False for r in weak), f"{len(weak)} weak assets")
    check("promoted share is a shortlist, not half the universe",
          len(out["promoted_candidates"]) <= out["fetched"] // 3,
          f"{len(out['promoted_candidates'])}/{out['fetched']}")

    check("data limitations stated", len(out["data_limitations"]) >= 3)
    check("method documents the no-persistence approach",
          "Recomputed" in out["method"]["historical_ranks"])
    check("promotion rules report the scan's volume bar",
          out["promotion_rules"]["volume_bar_this_scan"] is not None)

    cs._frame_cache.clear()
    full = cs.scan_universe(list(U.keys()), detail="full")
    check("full detail includes the per-asset block", "assets" in full)
    check("full detail covers every asset", len(full["assets"]) == full["fetched"])
    check("summary payload is far smaller than full",
          len(json.dumps(out)) * 3 < len(json.dumps(full)),
          f"summary={len(json.dumps(out))}B full={len(json.dumps(full))}B")
    check("bad detail value refused", "error" in cs.scan_universe(["BTC"], detail="verbose"))
    print(f"  promoted: {[p['symbol'] for p in out['promoted_candidates']]}")
    print(f"  suppressed: {list((out['promotion_rules']['suppressed_categories'] or {}).keys())}")
    print(f"  breadth:  {out['market_breadth']['classification']}"
          f" / {out['market_breadth']['rotation']}")

    cs._frame_cache.clear()
    no_btc = cs.scan_universe(["A0", "A1"])
    check("missing BTC aborts rather than returning partial results",
          "error" in no_btc and "BTC" in no_btc["error"])
finally:
    ds.fetch_ohlcv = real_fetch

print("\n=== 11. input guardrails ===")
check("empty universe refused", "error" in cs.scan_universe([]))
check("oversized universe refused", "error" in cs.scan_universe(["BTC"] * 61))

print("\n=== top 5 by composite percentile ===")
rows = [(s, d["composite_percentile"], d["rank_change_3d"], d["rank_acceleration"])
        for s, d in A.items() if d.get("rankable")]
rows.sort(key=lambda r: r[1] if r[1] is not None else -1, reverse=True)
for s, p, c, a in rows[:5]:
    print(f"  {s:6} percentile={p:6.1f}  3d_change={c:7.2f}  acceleration={a:7.2f}")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
