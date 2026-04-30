import pandas as pd
import numpy as np
import os

def create_feature_and_append(candle_data, file_path=r"C:\project\stock market\code\core\history.csv"):
    # 1. تحميل آخر 50 سطر فقط (لأطول نافذة حسابية عندك وهي الـ EMA-50)
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        # نأخذ آخر 50 سطر لنتمكن من حساب المتوسطات والمؤشرات
        last_rows = df_old.tail(50).copy()
        last_rows['date'] = pd.to_datetime(last_rows['date'], format="mixed")
    else:
        print("❌ Error: History file not found!")
        return

    # 2. تجهيز بيانات الشمعة الجديدة كسطر
    new_date = pd.to_datetime(candle_data["timestamp"], unit="ms")
    new_row_data = {
        "date": new_date,
        "open": float(candle_data["open"]),
        "high": float(candle_data["high"]),
        "low": float(candle_data["low"]),
        "close": float(candle_data["close"]),
        "volume": float(candle_data["volume"]),
    }
    
    # تحويل السطر لـ DataFrame مؤقت للدمج
    current_df = pd.DataFrame([new_row_data])
    
    # دمج آخر البيانات مع السطر الجديد للحسابات
    temp_df = pd.concat([last_rows, current_df], ignore_index=True)
    
    # 3. حساب الـ Features للسطر الأخير فقط (Index -1)
    # ملاحظة: سنستخدم نفس المنطق لكن سنستخرج القيمة الأخيرة فقط
    
    # --- Time Features ---

    temp_df.loc[temp_df.index[-1], "day_of_week"]  = new_date.dayofweek
    temp_df.loc[temp_df.index[-1], "month"]        = new_date.month
    temp_df.loc[temp_df.index[-1], "week_of_year"] = int(new_date.isocalendar()[1])
    temp_df.loc[temp_df.index[-1], "is_month_end"] = int(new_date.is_month_end)

    # --- Returns & Momentum ---
    # نستخدم iloc[-2] للوصول للسعر السابق
    prev_close = temp_df["close"].iloc[-2]
    curr_close = temp_df["close"].iloc[-1]
    
    temp_df.loc[temp_df.index[-1], "return_1"]   = (curr_close / prev_close) - 1
    temp_df.loc[temp_df.index[-1], "log_return"] = np.log(curr_close / prev_close)
    
    # Rolling Mean (آخر 7 و 14 سطر)
    temp_df.loc[temp_df.index[-1], "return_mean_7"]  = temp_df["log_return"].tail(7).mean()
    temp_df.loc[temp_df.index[-1], "return_mean_14"] = temp_df["log_return"].tail(14).mean()
    
    # Momentum
    temp_df.loc[temp_df.index[-1], "momentum_7"]  = (curr_close / temp_df["close"].iloc[-8]) - 1
    temp_df.loc[temp_df.index[-1], "momentum_14"] = (curr_close / temp_df["close"].iloc[-15]) - 1

    # --- Volatility ---
    temp_df.loc[temp_df.index[-1], "volatility_7"]  = temp_df["log_return"].tail(7).std()
    temp_df.loc[temp_df.index[-1], "volatility_30"] = temp_df["log_return"].tail(30).std()
    
    vol7 = temp_df.loc[temp_df.index[-1], "volatility_7"]
    vol30 = temp_df.loc[temp_df.index[-1], "volatility_30"]
    temp_df.loc[temp_df.index[-1], "vol_regime"] = vol7 / (vol30 + 1e-9)

    # --- RSI ---
    delta = temp_df["close"].diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    rs = gain / (loss + 1e-9)
    temp_df.loc[temp_df.index[-1], "rsi_14"] = 100 - (100 / (1 + rs))

    # --- EMA (حساب يدوي للـ EMA الجديد بناءً على القديم لتوفير الوقت) ---
    def calc_ema(new_val, prev_ema, span):
        alpha = 2 / (span + 1)
        return (new_val * alpha) + (prev_ema * (1 - alpha))

    temp_df.loc[temp_df.index[-1], "ema_20"] = calc_ema(curr_close, temp_df["ema_20"].iloc[-2], 20)
    temp_df.loc[temp_df.index[-1], "ema_50"] = calc_ema(curr_close, temp_df["ema_50"].iloc[-2], 50)

    # --- ATR ---
    high, low, prev_c = temp_df["high"].iloc[-1], temp_df["low"].iloc[-1], temp_df["close"].iloc[-2]
    tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
    # تقريب للـ ATR الجديد
    temp_df.loc[temp_df.index[-1], "atr_14"] = (temp_df["atr_14"].iloc[-2] * 13 + tr) / 14

    # --- الـ target و time_idx ---
    temp_df.loc[temp_df.index[-1], "time_idx"] = temp_df["time_idx"].iloc[-2] + 1
    # الـ target للسطر السابق أصبح الآن معروفاً وهو log_return الحالي
    temp_df.loc[temp_df.index[-2], "target"] = temp_df.loc[temp_df.index[-1], "log_return"]

    # 4. حفظ السطر الأخير فقط في الملف
    # ملاحظة: نقوم بتحديث السطر قبل الأخير (بسبب الـ target) وإضافة السطر الجديد
    final_row = temp_df.iloc[[-1]]
    
    # لتحديث الـ target في السطر السابق، يجب إعادة كتابة الملف أو التعامل مع التخزين بشكل أذكى
    # هنا سنضيف السطر الجديد فقط (الـ target للسطر الحالي سيبقى NaN حتى تأتي الشمعة القادمة)
    final_row.to_csv(file_path, mode='a', index=False, header=False)
    
    print(f"✅ Fast Append Done: {new_date}")