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

