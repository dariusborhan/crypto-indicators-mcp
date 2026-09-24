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
| `suggest_position_size(account_equity_usd, entry_price, collar_adjusted_stop_price, ...)` | Position size from a risk budget, not a chosen dollar amount. |
| `validate_trade_setup(entry_price, stop_trigger_price, target_price, ...)` | Hard gate a proposed entry must clear: reward-to-risk and sizing arithmetic computed once, not in prose. |
| `get_correlation_matrix(symbols, lookback_days)` | Pairwise and to-BTC return correlation, for the portfolio diversification rule. |
| `get_liquidity_profile(symbol, depth)` | Live order-book spread and depth snapshot (0.5%/1%/2% bands), to catch assets whose indicators compute but whose venue barely trades them. |
| `get_regime(symbol, timeframe)` | Trend/chop/high-vol label plus how many bars it has persisted — `get_market_context` has no memory of this by itself. |
| `detect_divergence(symbol, timeframe)` | Regular bullish/bearish divergence between price and RSI/MACD at the last two confirmed swing points. |
| `scan_universe(symbols)` | Cross-sectional pass over the whole eligible universe: relative-strength percentile per asset, how that percentile has moved over 1 and 3 days and whether it is accelerating, each asset's volatility state, a market breadth snapshot, and a promoted-candidate shortlist with its reasons. Promotion selects assets for a deep dive; it is never a signal and relaxes no gate. |
| `list_tradeable_symbols(search)` | Symbols with market data available. |

### Indicators computed

Trend and SMA/EMA 20/50/200 with MA alignment · RSI(14) · Stochastic RSI
(14,14,3,3) · MACD(12/26/9) with crossover detection · Bollinger Bands (20, 2
std) with %B, bandwidth, and squeeze detection · ATR(14) including ATR as a
percentage of price · volume vs its 20-period average · confirmed swing
highs/lows and market structure · nearest support and resistance · Fibonacci
retracement levels · relative strength vs a benchmark · multi-timeframe
confluence (1w/1d/4h/1h) with a numeric confluence score · cross-asset return
correlation · order-book spread and depth · trend/chop/high-vol regime with
duration · RSI/MACD divergence.

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

## Quantitative discovery layer (v2)

`scan_universe()` now separates **candidate discovery** from **trade permission**. In addition to cross-sectional relative strength, rank trajectory, volatility state, and breadth, it reports:

- prior-baseline return, volume, and range anomaly z-scores;
- 20-day vs 60-day BTC correlation and correlation delta;
- a transparent 0-100 `candidate_priority` decomposed into leadership, emergence, participation, volatility-transition, and independence dimensions;
- `recommended_deep_dives`, the top three non-BTC candidates on every healthy full scan.

`recommended_deep_dives` is deliberately **not a buy list**. Its purpose is to prevent the orchestration model from subjectively deciding that nothing is worth investigating. Every existing technical, reward-to-risk, liquidity, execution, churn, correlation, drawdown, and portfolio-risk gate still applies after the deep dive.

The trade-math defaults are aligned to the current strategy: 2% normal per-position risk budget and 10% total open-risk cap. Drawdown-band overrides remain explicit caller inputs.
