from dataclasses import dataclass, field


@dataclass
class Recommendation:
    action: str  # "BUY" | "HOLD" | "SELL"
    score: int
    reasons: list[str] = field(default_factory=list)


def recommend(
    predicted_return: float,
    ci_lower: float,
    ci_upper: float,
    rsi: float,
    macd_hist: float,
    return_threshold: float = 0.01,
) -> Recommendation:

    score = 0
    reasons = []

    if predicted_return > return_threshold and ci_lower > 0:
        score += 1
        reasons.append(f"Predicted return {predicted_return:+.2%} is positive and the confidence interval stays above zero.")
    elif predicted_return < -return_threshold and ci_upper < 0:
        score -= 1
        reasons.append(f"Predicted return {predicted_return:+.2%} is negative and the confidence interval stays below zero.")
    else:
        reasons.append("Forecast direction is not confidently positive or negative (confidence interval crosses zero).")

    if rsi > 70:
        score -= 1
        reasons.append(f"RSI at {rsi:.1f} signals overbought conditions.")
    elif rsi < 30:
        score += 1
        reasons.append(f"RSI at {rsi:.1f} signals oversold conditions.")

    if macd_hist > 0:
        score += 1
        reasons.append("MACD histogram is positive (bullish momentum).")
    elif macd_hist < 0:
        score -= 1
        reasons.append("MACD histogram is negative (bearish momentum).")

    if score > 0:
        action = "BUY"
    elif score < 0:
        action = "SELL"
    else:
        action = "HOLD"

    return Recommendation(action=action, score=score, reasons=reasons)


def project_portfolio_value(investment_amount: float, current_price: float, predicted_price: float) -> dict:
    """Estimates a future portfolio value from a predicted future price,
    assuming the investment is fully deployed into whole+fractional shares
    at the current price today.

    Returns share count, projected value, and absolute/percentage gain.
    """
    if current_price <= 0:
        raise ValueError("current_price must be positive")

    shares = investment_amount / current_price
    projected_value = shares * predicted_price
    gain = projected_value - investment_amount
    gain_pct = gain / investment_amount if investment_amount else 0.0

    return {
        "shares": shares,
        "projected_value": projected_value,
        "gain": gain,
        "gain_pct": gain_pct,
    }
