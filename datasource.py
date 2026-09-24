"""
Market data via Kraken's public REST API.

Why Kraken:
  - Native 4-hour (240m) and 1-day (1440m) candle intervals, which is exactly
    what the strategy spec calls for ("At minimum consider the 1-day and
    4-hour charts").
  - Returns up to 720 candles per request, enough for a genuine 200-period
    moving average on both timeframes.
  - No API key, no account, no signup.
  - Accessible from the US.
  - Real exchange volume rather than an aggregator's estimate.

Two details that matter for correctness:

  1. THE LAST CANDLE IS INCOMPLETE. Kraken's final row is the currently
     forming candle. Computing a daily RSI that includes a candle three hours
     into its 24-hour window produces a number that disagrees with every
     charting platform and flickers minute to minute. By default the
     incomplete candle is dropped from indicator computation and reported
     separately as the live price.

  2. KRAKEN USES NON-STANDARD ASSET CODES. Bitcoin is XBT, not BTC, and pair
     names are things like XXBTZUSD. Rather than hardcoding a mapping that
     will drift, pairs are resolved at runtime from the AssetPairs endpoint
     and cached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

KRAKEN_BASE = "https://api.kraken.com/0/public"
USER_AGENT = "crypto-indicators-mcp/1.0"
REQUEST_TIMEOUT = 30.0

# Kraken OHLC intervals, in minutes.
INTERVALS: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}

# Seconds per interval, used to decide whether the final candle is still open.
INTERVAL_SECONDS: dict[str, int] = {k: v * 60 for k, v in INTERVALS.items()}

# Quote currencies to try, in order of preference.
QUOTE_PREFERENCE = ["ZUSD", "USD", "USDT", "USDC", "ZEUR", "EUR"]


class DataSourceError(RuntimeError):
    """Raised when market data cannot be retrieved or trusted."""


@dataclass
class OHLCVResult:
    symbol: str
    kraken_pair: str
    timeframe: str
    df: pd.DataFrame          # completed candles only
    live_price: float | None  # from the in-progress candle, if there was one
    dropped_incomplete: bool
    last_closed_time: str | None
    fetched_at: str


@dataclass
class OrderBookResult:
    symbol: str
    kraken_pair: str
    bids: list[tuple[float, float]]  # (price, volume), best first
    asks: list[tuple[float, float]]  # (price, volume), best first
    fetched_at: str


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(path: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict:
    """GET a Kraken public endpoint, with retry on rate limiting."""
    url = f"{KRAKEN_BASE}/{path}"
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code == 429:
            time.sleep(3.0 * (attempt + 1))
            last_err = DataSourceError("Kraken rate limit (HTTP 429)")
            continue

        if resp.status_code != 200:
            raise DataSourceError(
                f"Kraken returned HTTP {resp.status_code} for {path}. "
                f"Body: {resp.text[:300]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise DataSourceError(
                f"Kraken returned a non-JSON response for {path}: {resp.text[:300]}"
            ) from exc

        errors = payload.get("error") or []
        if errors:
            joined = "; ".join(str(e) for e in errors)
            if any("Rate limit" in str(e) for e in errors):
                time.sleep(3.0 * (attempt + 1))
                last_err = DataSourceError(f"Kraken rate limit: {joined}")
                continue
            raise DataSourceError(f"Kraken API error for {path}: {joined}")

        if "result" not in payload:
            raise DataSourceError(
                f"Kraken response for {path} had no 'result' field: {str(payload)[:300]}"
            )
        return payload["result"]

    raise DataSourceError(
        f"Failed to fetch {path} after {retries} attempts. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Pair resolution
# ---------------------------------------------------------------------------

_pair_cache: dict[str, Any] | None = None
_pair_cache_time: float = 0.0
_PAIR_CACHE_TTL = 3600.0


def _load_pairs(force: bool = False) -> dict[str, Any]:
    global _pair_cache, _pair_cache_time
    now = time.time()
    if not force and _pair_cache is not None and now - _pair_cache_time < _PAIR_CACHE_TTL:
        return _pair_cache
    result = _get("AssetPairs")
    _pair_cache = result
    _pair_cache_time = now
    return result


# Kraken assigns a handful of assets a wsname/base ticker that differs from
# the ticker used everywhere else (brokerages, CoinGecko, etc). Dogecoin is
# the clear case: Kraken's own wsname is still "XDG/USD", not "DOGE/USD", so
# without this table DOGE silently fails to resolve on every call. Extend
# this table as other mismatches turn up (run every Robinhood symbol through
# resolve_pair() and log failures to find them).
_SYMBOL_ALIASES: dict[str, str] = {
    "DOGE": "XDG",
}
_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in _SYMBOL_ALIASES.items()}


def _normalize_base(symbol: str) -> str:
    s = symbol.strip().upper()
    # Kraken calls Bitcoin XBT. Accept the name everyone else uses.
    if s in ("BTC", "XBT"):
        return "XBT"
    return _SYMBOL_ALIASES.get(s, s)


def resolve_pair(symbol: str) -> str:
    """
    Map a plain symbol ("BTC", "ETH", "SOL") to a Kraken pair name.

    Prefers USD quotes, and prefers pairs that are actively tradeable. Raises
    DataSourceError with the closest matches if nothing is found, rather than
    guessing a pair name.
    """
    want_base = _normalize_base(symbol)
    pairs = _load_pairs()

    candidates: list[tuple[int, str]] = []
    for name, info in pairs.items():
        if not isinstance(info, dict):
            continue
        if info.get("status") not in (None, "online"):
            continue
        base = str(info.get("base", "")).upper()
        quote = str(info.get("quote", "")).upper()
        wsname = str(info.get("wsname", "")).upper()

        base_match = (
            base == want_base
            or base == f"X{want_base}"
            or base.lstrip("X") == want_base
            or wsname.split("/")[0] == want_base
        )
        if not base_match:
            continue
        if quote not in QUOTE_PREFERENCE:
            continue
        # Skip derivative/multiplier pairs such as ETH2.S or staked variants
        if "." in base or "." in name:
            continue
        candidates.append((QUOTE_PREFERENCE.index(quote), name))

    if not candidates:
        near = sorted(
            {
                str(i.get("wsname", n))
                for n, i in pairs.items()
                if isinstance(i, dict) and want_base in str(i.get("wsname", "")).upper()
            }
        )[:10]
        hint = f" Similar pairs on Kraken: {near}" if near else ""
        raise DataSourceError(
            f"No Kraken USD pair found for symbol '{symbol}'. "
            f"The asset may not be listed on Kraken, or may use a different "
            f"ticker there.{hint}"
        )

    candidates.sort()
    return candidates[0][1]


def list_symbols(search: str | None = None, limit: int = 60) -> list[dict[str, str]]:
    """Tradeable USD-quoted pairs on Kraken, optionally filtered."""
    pairs = _load_pairs()
    out: list[dict[str, str]] = []
    for name, info in pairs.items():
        if not isinstance(info, dict):
            continue
        if info.get("status") not in (None, "online"):
            continue
        quote = str(info.get("quote", "")).upper()
        if quote not in ("ZUSD", "USD"):
            continue
        wsname = str(info.get("wsname", name))
        base = wsname.split("/")[0] if "/" in wsname else str(info.get("base", ""))
        display = "BTC" if base in ("XBT", "XXBT") else _REVERSE_ALIASES.get(base.upper(), base)
        if search and search.strip().upper() not in display.upper():
            continue
        out.append({"symbol": display, "kraken_pair": name, "wsname": wsname})
    out.sort(key=lambda d: d["symbol"])
    return out[:limit]


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    drop_incomplete: bool = True,
) -> OHLCVResult:
    """
    Fetch OHLCV candles for `symbol` at `timeframe`.

    The final candle returned by Kraken is the one currently forming. Unless
    `drop_incomplete` is False it is removed from the frame used for indicator
    computation and surfaced separately as `live_price`.
    """
    if timeframe not in INTERVALS:
        raise DataSourceError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {sorted(INTERVALS)}"
        )

    pair = resolve_pair(symbol)
    result = _get("OHLC", {"pair": pair, "interval": INTERVALS[timeframe]})

    rows = None
    for key, value in result.items():
        if key == "last":
            continue
        if isinstance(value, list):
            rows = value
            break
    if rows is None:
        raise DataSourceError(
            f"Kraken OHLC response for {pair} contained no candle series. "
            f"Keys present: {list(result.keys())}"
        )
    if not rows:
        raise DataSourceError(f"Kraken returned zero candles for {pair} at {timeframe}.")

    # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    )
    for col in ("open", "high", "low", "close", "vwap", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "time"])
    if len(df) < before:
        # Malformed rows are dropped rather than filled. Never interpolate price.
        pass
    if df.empty:
        raise DataSourceError(f"All candles for {pair} at {timeframe} failed validation.")

    df = df.sort_values("time").reset_index(drop=True)

    live_price: float | None = None
    dropped = False
    now = time.time()
    span = INTERVAL_SECONDS[timeframe]
    last_open = float(df["time"].iloc[-1])
    if drop_incomplete and (now - last_open) < span and len(df) > 1:
        live_price = float(df["close"].iloc[-1])
        df = df.iloc[:-1].reset_index(drop=True)
        dropped = True

    last_closed = (
        pd.to_datetime(float(df["time"].iloc[-1]), unit="s", utc=True).isoformat()
        if not df.empty
        else None
    )

    return OHLCVResult(
        symbol=_display_symbol(symbol),
        kraken_pair=pair,
        timeframe=timeframe,
        df=df[["open", "high", "low", "close", "volume"]].copy(),
        live_price=live_price,
        dropped_incomplete=dropped,
        last_closed_time=last_closed,
        fetched_at=pd.Timestamp.utcnow().isoformat(),
    )


def _display_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return "BTC" if s in ("XBT", "BTC") else s


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

def fetch_order_book(symbol: str, depth: int = 100) -> OrderBookResult:
    """
    Fetch a live order-book snapshot for `symbol` from Kraken's public Depth
    endpoint.

    This is a single point-in-time snapshot, not a time series -- it reflects
    resting orders at the moment of the request and can change materially a
    second later. It has no visibility into futures order books, off-exchange
    liquidity, or hidden/iceberg orders.
    """
    pair = resolve_pair(symbol)
    result = _get("Depth", {"pair": pair, "count": depth})

    book = None
    for value in result.values():
        if isinstance(value, dict) and "bids" in value and "asks" in value:
            book = value
            break
    if book is None:
        raise DataSourceError(
            f"Kraken Depth response for {pair} contained no order book. "
            f"Keys present: {list(result.keys())}"
        )

    def _levels(raw: list) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for row in raw:
            try:
                price, volume = float(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if price > 0 and volume > 0:
                out.append((price, volume))
        return out

    bids = _levels(book.get("bids", []))
    asks = _levels(book.get("asks", []))
    if not bids or not asks:
        raise DataSourceError(f"Kraken returned an empty order book for {pair}.")

    return OrderBookResult(
        symbol=_display_symbol(symbol),
        kraken_pair=pair,
        bids=bids,
        asks=asks,
        fetched_at=pd.Timestamp.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose() -> dict[str, Any]:
    """Connectivity and data-shape self-test, for running on the host machine."""
    report: dict[str, Any] = {"source": "Kraken public REST API", "checks": []}

    def record(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"check": name, "ok": ok, "detail": detail})

    try:
        t = _get("Time")
        record("reachability", True, f"Kraken server time: {t.get('rfc1123', t)}")
    except Exception as exc:  # noqa: BLE001
        record("reachability", False, f"{type(exc).__name__}: {exc}")
        report["overall"] = "FAILED - cannot reach Kraken"
        return report

    try:
        pairs = _load_pairs(force=True)
        record("asset_pairs", True, f"{len(pairs)} pairs listed")
    except Exception as exc:  # noqa: BLE001
        record("asset_pairs", False, f"{type(exc).__name__}: {exc}")
        report["overall"] = "FAILED - cannot list pairs"
        return report

    for sym in ("BTC", "ETH"):
        try:
            p = resolve_pair(sym)
            record(f"resolve:{sym}", True, f"-> {p}")
        except Exception as exc:  # noqa: BLE001
            record(f"resolve:{sym}", False, f"{type(exc).__name__}: {exc}")

    for tf in ("1d", "4h"):
        try:
            r = fetch_ohlcv("BTC", tf)
            enough = len(r.df) >= 200
            record(
                f"ohlcv:BTC:{tf}",
                True,
                (
                    f"{len(r.df)} completed candles "
                    f"({'enough' if enough else 'NOT enough'} for a 200-period MA); "
                    f"last close {r.last_closed_time}; "
                    f"incomplete candle dropped: {r.dropped_incomplete}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            record(f"ohlcv:BTC:{tf}", False, f"{type(exc).__name__}: {exc}")

    failed = [c for c in report["checks"] if not c["ok"]]
    report["overall"] = "OK" if not failed else f"{len(failed)} check(s) failed"
    return report
