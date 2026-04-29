from socket import close

import numpy as np

# breakout
# trend following
# mean reversion



def strategy_decision(
    ema_20,
    ema_50,
    z_score,
    resistance_20,
    support_20,
    trend_conf,
    mean_conf,
    breakout_conf,
    close
):

    # -------------------------
    # TREND
    # -------------------------
    buy_trend = trend_conf * (ema_20 > ema_50)
    sell_trend = trend_conf * (ema_20 < ema_50)

    # -------------------------
    # MEAN REVERSION
    # -------------------------
    buy_mean = mean_conf * (z_score < -2)
    sell_mean = mean_conf * (z_score > 2)

    # -------------------------
    # BREAKOUT
    # -------------------------
    buy_break = breakout_conf * (close > resistance_20)
    sell_break = breakout_conf * (close < support_20)

    # -------------------------
    # PROBABILITIES
    # -------------------------
    buy_prob = (buy_trend + buy_mean + buy_break) / 3
    sell_prob = (sell_trend + sell_mean + sell_break) / 3

    # ⚠️ FIX مهم (HOLD لازم يكون “residual” وليس 1 - sum مباشرة)
    hold_prob = max(0.0, 1 - (buy_prob + sell_prob))

    total = buy_prob + sell_prob + hold_prob + 1e-9

    buy_prob /= total
    sell_prob /= total
    hold_prob /= total

    probs = {
        "BUY": buy_prob,
        "SELL": sell_prob,
        "HOLD": hold_prob
    }

    action = max(probs, key=probs.get)
    confidence = probs[action]

    return {
        "action": action,
        "confidence": probs[action],
        "probs": probs
    }