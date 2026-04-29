from core.candle_builder import CandleBuilder
from agents.stream_agent import StreamAgent
from agents.feature_agent import FeatureAgent
from agents.ta_agent import TechnicalAnalysisAgent

from agents.strategy_agent import strategy_decision

def on_candle_close(symbol, candle):

    print(symbol, candle)
    print("🔥 candle closed")

    # 1. update feature memory
    feature_agent.update(candle)

    # 2. compute features
    features = feature_agent.compute_features()

    if features is None:
        return
    strategy_inputs = {
    "ema_20": features["ema_20"],
    "ema_50": features["ema_50"],
    "z_score": features["z_score_20"],
    "close": features["close"],
    "resistance_20": features["resistance_20"],
    "support_20": features["support_20"],
    "trend_conf": features["trend_conf"],
    "mean_conf": features["mean_conf"],
    "breakout_conf": features["breakout_conf"],
}
    

    decision = strategy_decision(**strategy_inputs)

    # 3. send to ML / strategy (لاحقاً)
    print(features)
    print("🔥 strategy decision:", decision)



builder = CandleBuilder(timeframe_seconds=60)

agent = StreamAgent(builder, on_candle_close)

# 🔥 هنا input مرة واحدة

feature_agent = FeatureAgent(
    history_path="core/history.parquet",
    symbol="BTCUSDT",
    window_days=30
)

# 🔥 أول تشغيل (bootstrapping)
feature_agent.bootstrap_history()
agent.run("BTCUSDT")