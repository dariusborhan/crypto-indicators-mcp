"""Deterministic smoke tests for the strategy-critical trade-math defaults."""
import indicators as ind


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name

# Current canonical normal-risk defaults: 2% per new position, 10% portfolio cap.
r = ind.validate_trade_setup(
    entry_price=100,
    stop_trigger_price=95,
    target_price=120,
    account_equity_usd=250,
    current_open_risk_usd=0,
)
check("default sizing uses 2% risk budget", r["position_size_detail"]["risk_usd"] <= 5.0001)

# Existing risk just under 10%; any positive new risk should breach the cap.
r2 = ind.validate_trade_setup(
    entry_price=100,
    stop_trigger_price=95,
    target_price=120,
    account_equity_usd=250,
    current_open_risk_usd=24.9,
)
check("default portfolio cap is 10%", "portfolio_risk_cap_exceeded" in r2["reasons"])

# Explicit drawdown-band override remains supported.
r3 = ind.validate_trade_setup(
    entry_price=100,
    stop_trigger_price=95,
    target_price=130,
    account_equity_usd=1000,
    current_open_risk_usd=0,
    risk_budget_pct=1.25,
    reward_to_risk_floor=2.0,
)
check("drawdown overrides remain accepted", r3["position_size_detail"]["risk_usd"] <= 12.5001)
print("All trade-math smoke tests passed.")
