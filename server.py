"""
crypto-indicators-mcp

An MCP server that gives a trading agent real, computed technical indicators
instead of guessed ones.

It pairs with a brokerage MCP (such as Robinhood's) that handles accounts and
order placement. This server does no trading, holds no credentials, and has no
write access to anything -- it only reads public market data and does math on
it.

Every response states where its numbers came from, how many candles they were
computed from, and which indicators could NOT be computed and why. That last
part is the point: an indicator that cannot be calculated from available
history is reported as unavailable, never approximated.
"""

from __future__ import annotations

import os
from typing import Any

   from mcp.server.fastmcp import FastMCP
   from mcp.server.transport_security import TransportSecuritySettings

   import datasource as ds
   import indicators as ind

   # FastMCP auto-enables "DNS rebinding protection" whenever it thinks it is
   # bound to localhost, and that protection only accepts a Host header of
   # literally "localhost" or "127.0.0.1". It exists to stop a malicious web page
   # from using a victim's browser to reach a dev server on their own machine --
   # a real concern for something running unauthenticated on a laptop, not for a
   # server deliberately deployed to a public domain. Left on its default here,
   # every genuine request from a hosted platform (whose Host header is the
   # platform's own domain, e.g. *.onrender.com) is rejected with HTTP 421
   # before it reaches any of this file's code.
   #
   # stateless_http=True means no per-client session state is held between
   # requests, so the process can be restarted, scaled, or woken from idle
   # without breaking an in-flight connection.
   mcp = FastMCP(
       "crypto-indicators",
       stateless_http=True,
       transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
   )

# Assets used to judge the broader market regime.
REGIME_ASSETS = ("BTC", "ETH")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unavailable_summary(bundle: dict[str, Any]) -> list[str]:
    """Collect human-readable reasons for every indicator that was not computed."""
    notes: list[str] = []
    for key, entry in bundle.items():
        if key == "bars_available" or not isinstance(entry, dict):
            continue
        if entry.get("available") is False and entry.get("reason"):
            notes.append(f"{key}: {entry['reason']}")
        elif entry.get("caution"):
            notes.append(f"{key}: {entry['caution']}")
    return notes


def _analyze_one(symbol: str, timeframe: str) -> dict[str, Any]:
    data = ds.fetch_ohlcv(symbol, timeframe)
    bundle = ind.compute_all(data.df)
    return {
        "symbol": data.symbol,
        "timeframe": timeframe,
        "source": "Kraken public API",
        "kraken_pair": data.kraken_pair,
        "completed_candles": len(data.df),
        "last_closed_candle_utc": data.last_closed_time,
        "live_price_in_progress_candle": data.live_price,
        "incomplete_candle_excluded": data.dropped_incomplete,
        "indicators": bundle,
        "data_limitations": _unavailable_summary(bundle),
        "fetched_at_utc": data.fetched_at,
    }


def _error(message: str, symbol: str | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "error": message,
        "indicators": None,
        "instruction_to_agent": (
            "Market data could not be retrieved or verified. Per the strategy "
            "rules, do not trade on unavailable or unverified data, and do not "
            "substitute estimated values."
        ),
    }
    if symbol:
        out["symbol"] = symbol
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def diagnose_data_feed() -> dict[str, Any]:
    """
    Run a connectivity and data-quality self-test against the market data
    source.

    Use this first, before relying on any other tool in this server, and again
    any time a tool returns an error. It verifies that the data source is
    reachable, that trading pairs can be resolved, and that enough candle
    history is being returned to support a 200-period moving average.
    """
    try:
        return ds.diagnose()
    except Exception as exc:  # noqa: BLE001
        return _error(f"Diagnostic failed: {type(exc).__name__}: {exc}")


@mcp.tool()
def list_tradeable_symbols(search: str = "") -> dict[str, Any]:
    """
    List USD-quoted crypto symbols with market data available.

    Args:
        search: Optional filter, e.g. "SOL". Leave empty to list the first
                60 alphabetically.

    Note: this lists what MARKET DATA is available for. Whether your brokerage
    supports trading a given asset is a separate question -- check that against
    the brokerage's own tools before forming a trade thesis.
    """
    try:
        syms = ds.list_symbols(search or None)
        return {
            "count": len(syms),
            "symbols": syms,
            "note": (
                "Availability of market data here does not imply your brokerage "
                "supports trading this asset. Verify tradeability separately."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(f"Could not list symbols: {type(exc).__name__}: {exc}")


@mcp.tool()
def get_indicators(symbol: str, timeframe: str = "1d") -> dict[str, Any]:
    """
    Compute the full technical indicator set for one asset on one timeframe.

    Returns trend and moving averages (SMA/EMA 20/50/200), RSI(14), MACD
    (12/26/9), ATR(14) including ATR as a percentage of price, volume relative
    to its 20-period average, confirmed swing highs/lows with market structure,
    nearest support and resistance, and Fibonacci retracement levels.

    Indicators that cannot be computed from the available history are returned
    with available=false and a reason. They are never estimated. Values
    computed from enough bars to exist but not enough to fully converge are
    flagged warmup_sufficient=false with a caution.

    The currently-forming candle is excluded from all calculations; its price
    is reported separately as live_price_in_progress_candle.

    Args:
        symbol: Asset ticker, e.g. "BTC", "ETH", "SOL".
        timeframe: One of 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w. Default "1d".
    """
    try:
        return _analyze_one(symbol, timeframe)
    except ds.DataSourceError as exc:
        return _error(str(exc), symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return _error(f"{type(exc).__name__}: {exc}", symbol=symbol)


@mcp.tool()
def analyze_asset(symbol: str) -> dict[str, Any]:
    """
    Full multi-timeframe analysis of one asset, in a single call.

    This is the primary tool for forming a trade thesis. It returns the
    complete indicator set on BOTH the 1-day and 4-hour timeframes, plus the
    asset's relative strength against BTC over the last 30 daily candles.

    Reviewing the daily for trend and the 4-hour for timing satisfies the
    multi-timeframe requirement directly. Read the "data_limitations" field on
    each timeframe before drawing conclusions -- it lists every indicator that
    could not be computed and why.

    Args:
        symbol: Asset ticker, e.g. "BTC", "ETH", "SOL".
    """
    out: dict[str, Any] = {"symbol": symbol.strip().upper(), "timeframes": {}}
    errors: list[str] = []

    for tf in ("1d", "4h"):
        try:
            out["timeframes"][tf] = _analyze_one(symbol, tf)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tf}: {type(exc).__name__}: {exc}")
            out["timeframes"][tf] = {"error": str(exc), "indicators": None}

    # Relative strength against BTC on the daily.
    try:
        if ds._normalize_base(symbol) == "XBT":
            out["relative_strength_vs_btc"] = {
                "available": False,
                "reason": "Asset is BTC; relative strength against itself is undefined.",
            }
        else:
            asset = ds.fetch_ohlcv(symbol, "1d")
            btc = ds.fetch_ohlcv("BTC", "1d")
            rs = ind.relative_strength(asset.df["close"], btc.df["close"], periods=30)
            out["relative_strength_vs_btc"] = rs.to_dict()
    except Exception as exc:  # noqa: BLE001
        out["relative_strength_vs_btc"] = {
            "available": False,
            "reason": f"Could not compute: {type(exc).__name__}: {exc}",
        }

    if errors:
        out["errors"] = errors
        out["instruction_to_agent"] = (
            "One or more timeframes failed to load. Do not form a trade thesis "
            "from partial data; either retry or decline to trade."
        )
    else:
        out["instruction_to_agent"] = (
            "Check 'data_limitations' on each timeframe before citing any "
            "indicator. Do not cite an indicator reported as unavailable."
        )
    return out


@mcp.tool()
def get_market_context() -> dict[str, Any]:
    """
    Assess the broader crypto market regime using BTC and ETH on the daily
    timeframe.

    Use this BEFORE entering any altcoin position, as the strategy requires.
    Returns each benchmark's trend, moving-average alignment, RSI, MACD, and
    ATR-based volatility, plus a plain-language read of the overall regime.
    """
    out: dict[str, Any] = {"timeframe": "1d", "benchmarks": {}}
    reads: list[str] = []

    for sym in REGIME_ASSETS:
        try:
            a = _analyze_one(sym, "1d")
            out["benchmarks"][sym] = a
            trend = a["indicators"]["trend"]["value"]
            align = trend.get("ma_alignment", "unavailable")
            rsi_entry = a["indicators"]["rsi_14"]
            rsi_val = rsi_entry.get("value") if rsi_entry.get("available") else None
            atr_entry = a["indicators"]["atr_14"]
            atr_pct = (
                atr_entry["value"]["atr_percent_of_price"]
                if atr_entry.get("available")
                else None
            )
            reads.append(
                f"{sym}: MA alignment {align}; "
                f"RSI {rsi_val if rsi_val is not None else 'unavailable'}; "
                f"daily ATR {f'{atr_pct}% of price' if atr_pct is not None else 'unavailable'}"
            )
        except Exception as exc:  # noqa: BLE001
            out["benchmarks"][sym] = {"error": f"{type(exc).__name__}: {exc}"}
            reads.append(f"{sym}: unavailable ({exc})")

    out["summary_lines"] = reads
    out["instruction_to_agent"] = (
        "Interpret the regime yourself from these figures rather than relying "
        "on a precomputed label. If either benchmark is unavailable, treat the "
        "market regime as unknown and size positions accordingly or stand down."
    )
    return out


@mcp.tool()
def compare_to_benchmark(
    symbol: str, benchmark: str = "BTC", periods: int = 30
) -> dict[str, Any]:
    """
    Measure an asset's performance against a benchmark over a number of daily
    candles.

    Args:
        symbol: Asset ticker to evaluate, e.g. "SOL".
        benchmark: Ticker to compare against. Default "BTC".
        periods: Number of daily candles to measure over. Default 30.
    """
    try:
        a = ds.fetch_ohlcv(symbol, "1d")
        b = ds.fetch_ohlcv(benchmark, "1d")
        rs = ind.relative_strength(a.df["close"], b.df["close"], periods=periods)
        return {
            "symbol": a.symbol,
            "benchmark": b.symbol,
            "timeframe": "1d",
            "source": "Kraken public API",
            "result": rs.to_dict(),
        }
    except ds.DataSourceError as exc:
        return _error(str(exc), symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return _error(f"{type(exc).__name__}: {exc}", symbol=symbol)


@mcp.tool()
def suggest_volatility_stop(
    symbol: str,
    entry_price: float = 0.0,
    atr_multiple: float = 2.0,
    timeframe: str = "1d",
) -> dict[str, Any]:
    """
    Calculate a volatility-scaled stop distance from ATR, rather than a flat
    percentage.

    The strategy calls for stops "appropriate to the asset's volatility rather
    than an arbitrary fixed percentage". This returns the ATR-derived stop
    distance and the resulting stop price for a long position.

    This is an input to a decision, not a decision. It does not account for
    support levels or market structure, which should also inform where the
    invalidation point sits.

    Args:
        symbol: Asset ticker.
        entry_price: Intended entry. If 0, the last closed price is used.
        atr_multiple: How many ATRs below entry to place the stop. Default 2.0.
        timeframe: Timeframe for the ATR calculation. Default "1d".
    """
    try:
        data = ds.fetch_ohlcv(symbol, timeframe)
        a = ind.atr(data.df["high"], data.df["low"], data.df["close"], 14)
        if not a.available:
            return {
                "symbol": data.symbol,
                "available": False,
                "reason": a.reason,
                "instruction_to_agent": (
                    "ATR could not be computed, so a volatility-scaled stop "
                    "cannot be derived. Do not substitute an arbitrary "
                    "percentage without saying so explicitly."
                ),
            }

        last_close = float(data.df["close"].iloc[-1])
        entry = float(entry_price) if entry_price and entry_price > 0 else last_close
        atr_val = float(a.value["atr"])
        distance = atr_val * float(atr_multiple)
        stop = entry - distance

        return {
            "symbol": data.symbol,
            "timeframe": timeframe,
            "available": True,
            "entry_price_used": entry,
            "entry_price_source": "provided" if entry_price else "last closed candle",
            "atr_14": atr_val,
            "atr_percent_of_price": a.value["atr_percent_of_price"],
            "atr_multiple": atr_multiple,
            "stop_distance": distance,
            "suggested_stop_price": stop if stop > 0 else None,
            "stop_distance_percent": round(distance / entry * 100.0, 4) if entry else None,
            "warmup_sufficient": a.warmup_sufficient,
            "instruction_to_agent": (
                "This is a volatility reference, not a complete invalidation "
                "level. Cross-check against nearest support and market "
                "structure from get_indicators before committing to a stop."
            ),
        }
    except ds.DataSourceError as exc:
        return _error(str(exc), symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return _error(f"{type(exc).__name__}: {exc}", symbol=symbol)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
# Two ways to run:
#
#   stdio  -- the host launches this file as a subprocess. Used by desktop MCP
#             clients that run servers locally.
#   http   -- this file runs as a web service with a public URL, which is how
#             a hosted connector reaches it. This is the default, since that is
#             what deployment platforms expect.
#
# Set MCP_TRANSPORT=stdio to force the former.


def _build_http_app():
    """
    Wrap the MCP app with a health check and optional bearer-token auth.

    On auth: this server holds no credentials and cannot place trades -- it
    reads public market data and does arithmetic. So an open endpoint is not a
    financial risk. It is still worth protecting, because an open URL can be
    used to burn through the upstream exchange's rate limit or the hosting
    plan's quota. Set MCP_AUTH_TOKEN to require a token; leave it unset to run
    open.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    app = mcp.streamable_http_app()
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "crypto-indicators-mcp",
                "mcp_endpoint": "/mcp",
                "auth_required": bool(token),
            }
        )

    async def root(_request: Request) -> PlainTextResponse:
        return PlainTextResponse(
            "crypto-indicators-mcp is running.\n"
            "Connect an MCP client to the /mcp path of this URL.\n"
        )

    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/", root, methods=["GET"]))

    if token:
        from starlette.middleware.base import BaseHTTPMiddleware

        class BearerTokenMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # Health and root stay open so the platform can probe them.
                if request.url.path in ("/health", "/"):
                    return await call_next(request)
                header = request.headers.get("authorization", "")
                supplied = (
                    header[7:].strip()
                    if header.lower().startswith("bearer ")
                    else request.headers.get("x-api-key", "").strip()
                )
                if supplied != token:
                    return JSONResponse(
                        {"error": "Unauthorized. Supply the bearer token."},
                        status_code=401,
                    )
                return await call_next(request)

        app.add_middleware(BearerTokenMiddleware)

    return app


# Module-level app object, so a platform can also start this with
#   uvicorn server:app --host 0.0.0.0 --port $PORT
app = _build_http_app() if os.environ.get("MCP_TRANSPORT", "http") != "stdio" else None


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "http").strip().lower()

    if transport == "stdio":
        mcp.run()
        return

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
