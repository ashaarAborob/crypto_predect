import threading
import pandas as pd
from langchain_core.runnables import RunnableLambda

# استيراد الوظائف من الملفات الخاصة بك
from core.create_history import create_history
from core.candle_builder import CandleBuilder
from agents.stream_agent import StreamAgent
from agents.feature_agent import create_feature_and_append
from agents.strategy_agent import strategy_decision
from agents.tft_agent import tft_model
from agents.ml_agent import ml_model_predict
from agents.final_decision_agent import get_final_consensus # تأكد من وجود الملف

# --- 1. الدوال الأساسية ---

def initialize_system(symbol: str):
    """بناء ملف التاريخ الأولي"""
    print(f"🚀 Initializing History for {symbol}...")
    create_history(symbol=symbol, window_days=100)
    return symbol 

def run_analysis_async(symbol, last_row):
    """
    دالة التحليل الموحدة - تعمل في Thread منفصل
    تجمع بين الاستراتيجية، XGBoost، والـ TFT مع حساب السكور النهائي
    """
    print(f"\n--- 🧠 Deep Analysis Started for {symbol} ---")
    
    try:
        # تحويل السطر لـ DataFrame للموديلات
        last_row_df = pd.DataFrame([last_row])

        # أ. تنفيذ استراتيجية التحليل الفني (Classic Strategy)
        strat_res = strategy_decision(
            ema_20         = last_row["ema_20"],
            ema_50         = last_row["ema_50"],
            z_score        = last_row["z_score_20"],
            resistance_20  = last_row["resistance_20"],
            support_20     = last_row["support_20"],
            trend_conf     = last_row["trend_conf"],
            mean_conf      = last_row["mean_conf"],
            breakout_conf  = last_row["breakout_conf"],
            close          = last_row["close"]
        )

        # ب. تنفيذ نموذج XGBoost (Machine Learning)
        ml_res = ml_model_predict(last_row_df)

        # ج. تنفيذ نموذج TFT (Temporal Fusion Transformer) - الأهم والأدق
        try:
            tft_res = tft_model()
        except Exception as e:
            print(f"⚠️ TFT Error: {e}")
            tft_res = None

        # د. استدعاء الـ Final Decision Agent (الذي يعطي TFT الوزن الأكبر)
        # يقوم هذا الوكيل بحساب السكور المرجح وإعطاء القرار النهائي
        final_output = get_final_consensus(ml_res, strat_res, tft_res)

        # --- الطباعة النهائية للنتائج ---
        print("-" * 30)
        print(f"🌲 XGBoost Signal  : {ml_res['decision'] if ml_res else 'N/A'}")
        print(f"🛠 Strategy Signal : {strat_res['action']}")
        print(f"🤖 TFT AI Signal   : {tft_res['action'] if tft_res else 'N/A'}")
        print("-" * 30)
        print(f"📊 Final Weighted Score : {final_output['score']}")
        print(f"🏁 FINAL DECISION       : {final_output['decision']} {'🚀' if final_output['decision'] != 'HOLD' else '⚖️'}")
        print("-" * 30)

    except Exception as e:
        print(f"⚠️ Critical Analysis Error: {e}")

def handle_new_candle(symbol, candle):
    """تجهيز البيانات عند إغلاق الشمعة وتفعيل التحليل"""
    last_row = create_feature_and_append(candle) 
    
    if last_row is None:
        return

    # تشغيل التحليل في Thread منفصل
    analysis_thread = threading.Thread(
        target=run_analysis_async, 
        args=(symbol, last_row)
    )
    analysis_thread.daemon = True 
    analysis_thread.start()

def start_realtime_stream(symbol: str):
    """بدء البث المباشر"""
    print(f"📡 Starting Stream for {symbol}...")
    cb = CandleBuilder(timeframe_seconds=60)
    agent = StreamAgent(candle_builder=cb, on_candle_close=handle_new_candle)
    agent.run(symbol)

# --- 2. بناء الـ LangChain Chain ---

chain = (
    RunnableLambda(initialize_system) | 
    RunnableLambda(start_realtime_stream)
)

# --- 3. التشغيل ---

if __name__ == "__main__":
    target_symbol = "BTCUSDT"
    try:
        chain.invoke(target_symbol)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")