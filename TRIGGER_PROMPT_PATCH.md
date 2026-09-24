# Claude scheduled-task trigger patch

Apply these edits to the current trigger prompt after deploying this code. They are required for the orchestration layer to use the new deterministic discovery output.

## 1. Full-scan universe screening

Replace the instruction that asks Claude to perform a subjective lightweight first pass across the universe with:

> On every FULL SCAN with a healthy indicator feed, call `scan_universe` once with the complete eligible universe, including BTC, using `detail="summary"`. Treat `recommended_deep_dives` as the deterministic investigation queue. Perform the complete 1w/1d/4h/1h deep dive on the top three non-BTC symbols returned there (or all returned symbols if fewer than three). Do not veto a recommended deep dive merely because RSI is high, price is near an upper Bollinger Band, or the model subjectively considers the market extended. Those facts may affect the subsequent thesis, but they do not cancel the deep dive. `recommended_deep_dives` is NOT permission to trade: every existing technical-thesis, market-context, reward-to-risk, execution-preview, liquidity, correlation, churn, drawdown, and portfolio-risk gate still applies.

## 2. Candidate-priority evidence

Add:

> For each deep-dive candidate, read and report `candidate_priority` and the available anomaly/correlation fields returned by `scan_universe`. Treat them as discovery evidence, not independent entry signals. Do not recompute ranks, z-scores, or correlations in prose.

## 3. Risk defaults

The deployed MCP now defaults to the current normal-risk values: `risk_budget_pct=2.0` and `portfolio_risk_cap_pct=10.0`. Continue passing these values explicitly from the trigger prompt so the run is auditable. In the 15%-25% closed-equity drawdown band, continue passing the strategy's reduced risk-budget and increased R:R-floor values explicitly.

## 4. Duplicate-order backstop bug

Replace the current symbol-only rule:

> If a recent order already exists for that symbol, do not place another.

with:

> Immediately before placing an order, inspect recent orders for the symbol and suppress the new order only when recent history indicates a possible duplicate of the SAME intended action (same side/purpose/order type and materially similar quantity/trigger where applicable). A recent order of a different purpose must not block the next required action. In particular, a just-filled entry buy must never block its required protective stop; a cancelled stop must never block its replacement; and a stop cancellation must never block the sell that intentionally follows it. If order history is ambiguous about whether the same intended action already landed, fail safe: do not duplicate it, reconcile the order state, and ensure an existing position ends the run protected.

## 5. Journal-read wording

Clarify the failed-read rule to say:

> If full journal validation fails, make no journal modification other than clearing the `run_lock` created by this run. Do not trade or overwrite state derived from the failed read.

## 6. Universe discovery must not go through `list_tradeable_symbols`

`list_tradeable_symbols` (`datasource.list_symbols`) has no pagination and silently caps at 60 results, alphabetically. On a raw Robinhood catalog of 90+ symbols it was truncating the eligible universe before `scan_universe` ever ran. `scan_universe` itself now accepts up to 120 symbols per call (raised from 60) and resolves availability itself via `fetch_failures`, so there is no reason to pre-filter with `list_tradeable_symbols` at all.

Replace steps 1-4 of "BUILD THE FULL UNIVERSE EACH RUN" with:

> 1. Call `get_currency_pairs` (or equivalent Robinhood tool) to get the full list of cryptocurrencies Robinhood actually lets this account trade. Record the raw count and the full base-symbol list (e.g. "BTC-USD" -> "BTC").
> 2. Apply the permitted-universe exclusions to that raw list, using explicit, restatable criteria — this is a judgment about asset class (stablecoin, meme/micro-cap, leveraged/derivative, not spot-tradeable) that only Robinhood's own catalog data can answer, so do it BEFORE involving the indicators connector at all. Name which specific criterion applied per exclusion. The result is the STRATEGY-ELIGIBLE LIST for this cycle. Include BTC even though it may be used primarily as the benchmark.
> 3. Call `scan_universe` ONCE, passing the entire strategy-eligible list as symbols, with `detail="summary"` and no separate availability pre-check first — do NOT call `list_tradeable_symbols` as a filtering step; it only returns a partial, alphabetically-truncated slice of what has Kraken data and will not reflect the true availability of most of the universe. `scan_universe` resolves each symbol against Kraken itself and reports exactly which ones it could not fetch, with a reason, in `fetch_failures`. It accepts up to 120 symbols per call, comfortably above the full size of a typical brokerage crypto catalog; if the strategy-eligible list somehow still exceeds that, `scan_universe` returns an explicit error naming the count and the limit rather than silently truncating — in that case drop the lowest-liquidity names to fit and record that as an additional, named exclusion criterion in the funnel report.
> 4. Read the response: `fetch_failures` is the "Lacked reliable Kraken/indicator data" bucket for this cycle — record each symbol with the reason the tool gave, don't guess at one. The symbols actually fetched (requested minus `fetch_failures`) are the FINAL ELIGIBLE UNIVERSE actually screened this cycle. Do not manually reproduce `scan_universe`'s ranks, z-scores, breadth, correlations, or candidate-priority arithmetic.

And update the funnel-reporting bullets to:

> - Robinhood catalog: \<raw count\>
> - Excluded by strategy rules, broken down by specific criterion: \<counts\>
> - Lacked reliable Kraken/indicator data: \<count, symbols + reason from scan_universe's fetch_failures\>
> - Final eligible universe (what scan_universe actually fetched): \<count\>
> - Recommended deep dives: \<count and symbols, with candidate_priority for each\>
> - Full deep dives actually completed: \<count and symbols\>

This patch also required a code change already included in this repo: `crosssection.scan_universe`'s `max_symbols` default was raised from 60 to 120 so a full post-exclusion universe fits in one call.

