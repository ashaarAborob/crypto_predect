import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os


# =========================================================
# 🔥 FEATURE ENGINE (ONE SOURCE OF TRUTH)
# =========================================================
class FeatureEngine:

    def __init__(self, coin_name="BTC"):
        self.coin_name = coin_name

    # =========================
    # FULL / INCREMENTAL FEATURES
    # =========================
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["coin_name"] = self.coin_name

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"]

        # ---------- TIME ----------
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        # ---------- RETURNS ----------
        df["log_return"] = np.log(close / close.shift(1))
        df["return_mean_7"] = df["log_return"].rolling(7).mean()
        df["return_mean_14"] = df["log_return"].rolling(14).mean()

        # ---------- MOMENTUM ----------
        df["momentum_7"] = close.pct_change(7)
        df["momentum_14"] = close.pct_change(14)

        # ---------- VOLATILITY ----------
        df["volatility_7"] = df["log_return"].rolling(7).std()
        df["volatility_14"] = df["log_return"].rolling(14).std()
        df["volatility_30"] = df["log_return"].rolling(30).std()

        df["vol_regime"] = df["volatility_7"] / (df["volatility_30"] + 1e-9)

        # ---------- LAGS ----------
        for lag in [1, 2, 3, 7, 14]:
            df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
            df[f"close_lag_{lag}"] = close.shift(lag)

        df["lag_momentum_1"] = df["return_lag_1"] * df["momentum_7"]

        # ---------- RSI ----------
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)

        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_momentum"] = df["rsi_14"] * df["momentum_7"]

        # ---------- STRUCTURE ----------
        df["high_low_range"] = (high - low) / (close + 1e-9)
        df["close_open_range"] = (close - open_) / (open_ + 1e-9)

        df["buy_pressure"] = (close - low) / (high - low + 1e-9)
        df["sell_pressure"] = (high - close) / (high - low + 1e-9)

        # ---------- VOLUME ----------
        df["volume_ma_7"] = volume.rolling(7).mean()
        df["volume_ratio"] = volume / (df["volume_ma_7"] + 1e-9)

        # ---------- EMA ----------
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["ema_diff"] = df["ema_20"] - df["ema_50"]

        # ---------- TARGET ----------
        df["target"] = df["log_return"].shift(-1)

        return df


# =========================================================
# 🔥 FEATURE AGENT
# =========================================================
class FeatureAgent:

    def __init__(self, history_path, symbol, window_days=100, coin_name="BTC"):

        self.history_path = history_path
        self.symbol = symbol
        self.window = window_days

        self.engine = FeatureEngine(coin_name)

        # ======================
        # LOAD OR CREATE HISTORY
        # ======================
        if os.path.exists(history_path):
            print("📂 Loading history...")
            self.df = pd.read_csv(history_path)
            self.df["date"] = self.df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            self.df = self.df.sort_values("date").reset_index(drop=True)
        else:
            print("⚠️ No history → building...")
            self.df = self.bootstrap_history()

    # =====================================================
    # BUILD FULL HISTORY (ONLY ONCE)
    # =====================================================
    def bootstrap_history(self):

        url = "https://api.binance.com/api/v3/klines"

        end = int(datetime.utcnow().timestamp() * 1000)
        start = int((datetime.utcnow() - timedelta(days=self.window)).timestamp() * 1000)

        data = requests.get(url, params={
            "symbol": self.symbol,
            "interval": "1d",
            "startTime": start,
            "endTime": end,
            "limit": 1000
        }).json()

        df = pd.DataFrame([{
            "date": "date": pd.to_datetime(d[0], unit="ms").floor("min"),
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5]),
        } for d in data])

        df = self.engine.add_features(df)

        df = df.dropna().reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))

        self.save(df)
        return df

    # =====================================================
    # STREAM UPDATE (ONLY NEW ROW)
    # =====================================================
    def update(self, candle):

        new_row = pd.DataFrame([{
            "date": pd.to_datetime(candle["timestamp"], unit="ms"),
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"]
        }])

        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # keep window
        self.df = self.df.tail(self.window + 200)

        # 🔥 recompute ONLY tail features (still using same engine)
        self.df = self.engine.add_features(self.df)

        last_row = self.df.iloc[-1]

        self.save(self.df)

        return last_row.to_dict()

    # =====================================================
    def save(self, df):
        df.to_csv(self.history_path, index=False)