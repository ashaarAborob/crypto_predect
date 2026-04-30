import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


def create_history(
    symbol: str      = "BTCUSDT",
    window_days: int = 100,
    coin_name: str   = "BTC",
    save_path: str   = r"C:\project\stock market\code - Copy\core\history.csv",
) -> pd.DataFrame:

    # ─────────────────────────────────────────────
    # 1. FETCH  — مرة وحدة بس ✅
    # ─────────────────────────────────────────────
    url   = "https://api.binance.com/api/v3/klines"
    end   = int(datetime.utcnow().timestamp() * 1000)
    start = int((datetime.utcnow() - timedelta(days=window_days)).timestamp() * 1000)

    response = requests.get(url, params={
        "symbol"   : symbol,
        "interval" : "1d",
        "startTime": start,
        "endTime"  : end,
        "limit"    : 1000,
    })

    if response.status_code != 200:
        raise RuntimeError(f"Binance API error: {response.text}")

    df = pd.DataFrame([{
        "date"  : pd.to_datetime(d[0], unit="ms").floor("min"),
        "open"  : float(d[1]),
        "high"  : float(d[2]),
        "low"   : float(d[3]),
        "close" : float(d[4]),
        "volume": float(d[5]),
    } for d in response.json()])

    df = df.sort_values("date").reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 2. BASE SERIES
    # ─────────────────────────────────────────────
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]
    # بدل التنسيق القديم، استخدم format='mixed' أو ISO8601
    # ─────────────────────────────────────────────
    # 3. TIME FEATURES (KNOWN_REALS)
    # ─────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("min").dt.tz_convert(None)
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # ─────────────────────────────────────────────
    # 4. RETURNS
    # ─────────────────────────────────────────────
    df["return_1"]   = close.pct_change()
    df["log_return"] = df["close"].transform(lambda x: np.log(x / x.shift(1)))  # ✅ transform + log return

    # ✅ على log_return (كانت على return_1)
    df["return_mean_7"]  = df["log_return"].rolling(7).mean()
    df["return_mean_14"] = df["log_return"].rolling(14).mean()

    # ─────────────────────────────────────────────
    # 5. MOMENTUM  (محتاج قبل lag_momentum_1)
    # ─────────────────────────────────────────────
    df["momentum_7"]     = close.pct_change(7)
    df["momentum_14"]    = close.pct_change(14)

    # ─────────────────────────────────────────────
    # 6. VOLATILITY
    # ─────────────────────────────────────────────
    df["volatility_7"]  = df["log_return"].rolling(7).std()
    df["volatility_14"] = df["log_return"].rolling(14).std()
    df["volatility_30"] = df["log_return"].rolling(30).std()

    # ✅ vol_7 / vol_30  (كانت vol_14)
    df["vol_regime"]         = df["volatility_7"] / (df["volatility_30"] + 1e-9)
    df["volatility_squeeze"] = df["volatility_7"] / (df["volatility_30"] + 1e-9)

    df["trend_strength"] = df["momentum_14"] * df["volatility_7"]

    # ─────────────────────────────────────────────
    # 7. LAGS  ✅ على log_return
    # ─────────────────────────────────────────────
    for lag in [1, 2, 3, 7, 14]:
        df[f"close_lag_{lag}"]  = close.shift(lag)
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)  # ✅ log_return

    # ✅ return_lag_1 * momentum_7  (كانت momentum_7.shift(1))
    df["lag_momentum_1"] = df["return_lag_1"] * df["momentum_7"]

    # ─────────────────────────────────────────────
    # 8. RSI
    # ─────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"]    = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi"]

    # ✅ rsi_14 * momentum_7  (كانت .diff())
    df["rsi_momentum"] = df["rsi_14"] * df["momentum_7"]

    # ─────────────────────────────────────────────
    # 9. PRICE STRUCTURE
    # ─────────────────────────────────────────────
    df["hl_range"]         = (high - low) / (close + 1e-9)
    df["body"]             = (close - open_).abs()
    df["high_low_range"]   = df["hl_range"]                       # alias للموديل
    df["close_open_range"] = (close - open_) / (open_ + 1e-9)    # alias للموديل

    df["buy_pressure"]  = (close - low)  / (high - low + 1e-9)
    df["sell_pressure"] = (high - close) / (high - low + 1e-9)

    # ─────────────────────────────────────────────
    # 10. VOLUME
    # ─────────────────────────────────────────────
    df["volume_ma_7"]   = volume.rolling(7).mean()
    df["volume_ma_14"]  = volume.rolling(14).mean()
    df["volume_change"] = volume.pct_change()
    df["volume_ratio"]  = volume / (df["volume_ma_7"] + 1e-9)

    # ─────────────────────────────────────────────
    # 11. BOLLINGER / Z-SCORE
    # ─────────────────────────────────────────────
    df["mean_20"]    = close.rolling(20).mean()
    df["std_20"]     = close.rolling(20).std()
    df["z_score_20"] = (close - df["mean_20"]) / (df["std_20"] + 1e-9)
    df["bb_upper"]   = df["mean_20"] + 2 * df["std_20"]
    df["bb_lower"]   = df["mean_20"] - 2 * df["std_20"]
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / (df["mean_20"] + 1e-9)

    # ─────────────────────────────────────────────
    # 12. EMA
    # ─────────────────────────────────────────────
    df["ema_20"]      = close.ewm(span=20, adjust=False).mean()
    df["ema_50"]      = close.ewm(span=50, adjust=False).mean()
    df["ema_diff"]    = df["ema_20"] - df["ema_50"]
    df["ema_ratio"]   = df["ema_20"] / (df["ema_50"] + 1e-9)
    df["ema20_slope"] = df["ema_20"].diff()
    df["ema50_slope"] = df["ema_50"].diff()

    # ─────────────────────────────────────────────
    # 13. SUPPORT / RESISTANCE
    # ─────────────────────────────────────────────
    df["resistance_20"]      = high.rolling(20).max()
    df["support_20"]         = low.rolling(20).min()
    df["breakout_up_dist"]   = (close - df["resistance_20"]) / (df["resistance_20"] + 1e-9)
    df["breakout_down_dist"] = (close - df["support_20"])    / (df["support_20"]    + 1e-9)

    # ─────────────────────────────────────────────
    # 14. ATR
    # ─────────────────────────────────────────────
    tr = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        )
    )
    df["atr_14"] = pd.Series(tr, index=df.index).ewm(alpha=1/14, adjust=False).mean()

    # ─────────────────────────────────────────────
    # 15. CONFIDENCE FEATURES
    # ─────────────────────────────────────────────
    df["trend_conf"]    = np.tanh(df["ema_diff"].abs())
    df["mean_conf"]     = np.tanh(df["z_score_20"].abs())
    df["breakout_conf"] = 1 / (1 + np.exp(-((close - df["resistance_20"]) / df["atr_14"])))

    # ─────────────────────────────────────────────
    # 16. REQUIRED BY TFT MODEL ✅
    # ─────────────────────────────────────────────
    #df["coin_name"] = coin_name                   # static_categorical
    df["target"]    = df["log_return"].shift(-1)  # target

    # ─────────────────────────────────────────────
    # 17. CLEAN  +  time_idx بعد dropna ✅
    # ─────────────────────────────────────────────
    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))           # ✅ بعد dropna

    # ─────────────────────────────────────────────
    # 18. SAVE & RETURN
    # ─────────────────────────────────────────────
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"💾 Saved → {save_path}")

    print(f"✅ History ready: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"   Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = create_history(symbol="BTCUSDT", coin_name="BTC", window_days=100)
    print(df)

    df.describe()