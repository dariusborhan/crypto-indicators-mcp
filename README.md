# crypto-indicators-mcp

Real technical indicators for a Claude trading agent.

This is the companion piece to Robinhood's trading connector. Robinhood's MCP
handles the account and places orders; it does not hand back the deep price
history needed to compute a 200-period moving average, an RSI, or an ATR. This
server fills that gap so the agent reasons over **computed** indicators rather
than approximated ones.

It does no trading, holds no credentials, and has no write access to anything.
It reads public market data and does math on it.

---

## The one design rule

An indicator that cannot be computed from the available history is reported as
**unavailable, with a reason**. It is never estimated, padded, or guessed.

This is the software counterpart to the line in your strategy doc:

> "If the data needed to compute a given indicator reliably isn't available,
> say so explicitly rather than approximating or estimating a value."

A fabricated RSI that *sounds* precise is worse than no RSI, because it will be
cited in a trade thesis as though it were real. Every response therefore
carries a `data_limitations` list naming exactly what could not be calculated.

---

## Setup

**See `DEPLOY.md`** — it walks through hosting this on the web so you can add
it to Claude by pasting a URL, exactly like the Robinhood connector. No
terminal required, and it keeps running when your computer is off.

Running it on the web is the right default here. A locally-run server only
exists while your computer is awake and the desktop app is open, which does not
suit an agent that should be reachable on its own schedule.

### Running it locally instead (optional)

If you do want it local — for development, say — it still speaks stdio:

```bash
pip install -r requirements.txt
MCP_TRANSPORT=stdio python server.py
```

and in a desktop MCP client's config:

```json
{
  "mcpServers": {
    "crypto-indicators": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/crypto-indicators-mcp/server.py"],
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

Both paths must be absolute.

### Verifying either way

Ask Claude to **run `diagnose_data_feed`** before trusting anything else:

```
overall: OK
  reachability   ok    Kraken server time: ...
  asset_pairs    ok    ~1000 pairs listed
  resolve:BTC    ok    -> XXBTZUSD
  ohlcv:BTC:1d   ok    719 completed candles (enough for a 200-period MA)
  ohlcv:BTC:4h   ok    719 completed candles (enough for a 200-period MA)
```

---

## Configuration

| Environment variable | Default | What it does |
|---|---|---|
| `MCP_TRANSPORT` | `http` | `http` to run as a web service, `stdio` to run as a local subprocess. |
| `PORT` | `8000` | Port to listen on. Most hosts set this for you. |
| `HOST` | `0.0.0.0` | Interface to bind. |
| `MCP_AUTH_TOKEN` | unset | If set, `/mcp` requires this as a bearer token. `/health` stays open. |

When running over HTTP the MCP endpoint is at **`/mcp`**, and there is a
`/health` endpoint that reports status without requiring auth.

---

## Tools

| Tool | What it does |
|---|---|
| `diagnose_data_feed()` | Connectivity and data-quality self-test. Run this first. |
| `analyze_asset(symbol)` | **Main tool.** Full indicator set on both 1d and 4h, plus relative strength vs BTC, in one call. |
| `get_indicators(symbol, timeframe)` | Full indicator set for one asset on one timeframe. |
| `get_market_context()` | BTC and ETH daily regime. Run before any altcoin entry. |
| `compare_to_benchmark(symbol, benchmark, periods)` | Relative performance over N daily candles. |
| `suggest_volatility_stop(symbol, entry_price, atr_multiple, timeframe)` | ATR-derived stop distance instead of a flat percentage. |
| `list_tradeable_symbols(search)` | Symbols with market data available. |

### Indicators computed

Trend and SMA/EMA 20/50/200 with MA alignment · RSI(14) · MACD(12/26/9) with
crossover detection · ATR(14) including ATR as a percentage of price · volume
vs its 20-period average · confirmed swing highs/lows and market structure ·
nearest support and resistance · Fibonacci retracement levels · relative
strength vs a benchmark.

---

## Three things worth knowing

**The forming candle is excluded.** Kraken's last row is the candle currently
in progress. A daily RSI that includes a candle three hours into its 24-hour
window disagrees with every charting platform and flickers minute to minute.
It is dropped from all calculations and reported separately as
`live_price_in_progress_candle`.

**Some values are flagged provisional.** RSI, ATR, MACD and EMAs are recursive
— each value is seeded by the first observation, whose influence decays over
time. Where there are enough bars to compute a value but not enough for that
seed to wash out, the result carries `warmup_sufficient: false` and a caution.
The threshold is derived from the actual decay rate, not a rule of thumb.

**Market data availability is not tradeability.** This server tells you what
Kraken has price history for. Whether Robinhood lets you trade that asset is a
separate question — check it against Robinhood's own tools before building a
thesis on something you cannot buy.

---

## Tests

```bash
python3 test_indicators.py      # 57 checks: indicator math
python3 test_server_offline.py  # 71 checks: data layer and MCP tools
```

`test_indicators.py` validates every formula two ways: against a plain-Python
loop implementation of the textbook definition, and against mathematical
invariants that must hold (RSI of a rising series is 100, ATR of a
constant-range series equals that range, MACD of a flat series is 0, and so
on). It also asserts the refuse-to-guess contract — that short history yields
`available: false` rather than a number.

`test_server_offline.py` stubs the network with fixtures shaped like real
Kraken responses and exercises everything downstream: pair resolution,
parsing, incomplete-candle handling, every MCP tool, and the error paths.

These were written and run in a sandbox with no access to exchange APIs, so
**live reachability was never tested** — that is what `diagnose_data_feed` is
for, and why it is the first tool in the list.

---

## Limitations

- **Not backtested.** Nothing here has been validated against historical
  performance. It computes indicators correctly; it makes no claim that acting
  on them is profitable.
- **Kraken's data, not Robinhood's.** Prices will differ slightly from your
  execution venue. Fine for trend and momentum, not for precise fill modelling.
- **No live-data validation.** See above.
- **Indicators are not signals.** This server deliberately does not emit
  buy/sell recommendations. It returns numbers; the agent does the reasoning,
  which is where your strategy rules apply.
