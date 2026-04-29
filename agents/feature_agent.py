#from bitsandbytes import features
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


class FeatureAgent:

    def __init__(self, history_path, symbol, window_days=50):

        self.history_path = history_path
        self.symbol = symbol
        self.window = window_days

        # load history
        try:
            self.df = pd.read_parquet(history_path)
        except:
            self.df = pd.DataFrame()

    # =========================
    # BOOTSTRAP HISTORY
    # =========================
    def bootstrap_history(self):

        if not self.df.empty:
            print("✅ history already exists")
            return

        print("⏳ fetching historical data...")

        url = "https://api.binance.com/api/v3/klines"

        end = int(datetime.utcnow().timestamp() * 1000)
        start = int((datetime.utcnow() - timedelta(days=self.window)).timestamp() * 1000)

        params = {
            "symbol": self.symbol,
            "interval": "1d",
            "startTime": start,
            "endTime": end,
            "limit": 1000
        }

        data = requests.get(url, params=params).json()

        self.df = pd.DataFrame([{
            "timestamp": d[0],
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5])
        } for d in data])

        self.save()

        print("✅ history loaded:", len(self.df))

    # =========================
    # SAVE
    # =========================
    def save(self):
        self.df.to_parquet(self.history_path, index=False)

    # =========================
    # STREAM UPDATE
    # =========================
    def update(self, candle):

        self.df = pd.concat([self.df, pd.DataFrame([candle])], ignore_index=True)

        self.df = self.df.tail(self.window + 50)

        self.save()

        return self.compute_features()

    # =========================
    # FEATURE ENGINE (ALL IN ONE)
    # =========================
    def compute_features(self):

        df = self.df.copy()

        if len(df) < 20:
            return None

        close_series = df["close"]
        high_series = df["high"]
        low_series = df["low"]
        volume_series = df["volume"]

        # -------------------------
        # RETURNS
        # -------------------------
        close = close_series.iloc[-1]

        return_1 = close_series.pct_change().iloc[-1]
        log_return = np.log(close_series / close_series.shift(1)).iloc[-1]

        return_3 = close_series.pct_change(3).iloc[-1]
        return_5 = close_series.pct_change(5).iloc[-1]

        # -------------------------
        # VOLATILITY
        # -------------------------
        return_series = close_series.pct_change()

        vol_7 = return_series.rolling(7).std().iloc[-1]
        vol_14 = return_series.rolling(14).std().iloc[-1]
        vol_30 = return_series.rolling(30).std().iloc[-1]

        vol_regime = vol_7 / (vol_30 + 1e-9)

        # -------------------------
        # MOMENTUM
        # -------------------------
        momentum_7 = (close_series / close_series.shift(7) - 1).iloc[-1]
        momentum_14 = (close_series / close_series.shift(14) - 1).iloc[-1]

        trend_strength = momentum_14 * vol_7

        # -------------------------
        # MOVING AVERAGES
        # -------------------------
        ma_10 = close_series.rolling(10).mean().iloc[-1]
        ma_20 = close_series.rolling(20).mean().iloc[-1]

        trend_ratio = ma_10 / (ma_20 + 1e-9)

        # -------------------------
        # PRICE STRUCTURE
        # -------------------------
        hl_range = ((high_series - low_series) / (close_series + 1e-9)).iloc[-1]
        body = abs(close_series.iloc[-1] - df["open"].iloc[-1])

        # -------------------------
        # VOLUME
        # -------------------------
        volume_change = volume_series.pct_change().iloc[-1]
        volume_ma_7 = volume_series.rolling(7).mean().iloc[-1]
        volume_ratio = volume_series.iloc[-1] / (volume_ma_7 + 1e-9)

        # -------------------------
        # RSI
        # -------------------------
        rsi = self._rsi(close_series, 14).iloc[-1]

        # -------------------------
        # BOLLINGER
        # -------------------------
        mean_20 = close_series.rolling(20).mean().iloc[-1]
        std_20 = close_series.rolling(20).std().iloc[-1]

        z_score_20 = (close - mean_20) / (std_20 + 1e-9)

        bb_upper = mean_20 + 2 * std_20
        bb_lower = mean_20 - 2 * std_20
        bb_width = (bb_upper - bb_lower) / (mean_20 + 1e-9)

        # -------------------------
        # EMA
        # -------------------------
        ema_20_series = close_series.ewm(span=20).mean()
        ema_50_series = close_series.ewm(span=50).mean()

        ema_20 = ema_20_series.iloc[-1]
        ema_50 = ema_50_series.iloc[-1]

        ema_diff = ema_20 - ema_50
        ema_ratio = ema_20 / (ema_50 + 1e-9)

        ema20_slope = ema_20_series.diff().iloc[-1]
        ema50_slope = ema_50_series.diff().iloc[-1]

        # -------------------------
        # SUPPORT / RESISTANCE
        # -------------------------
        resistance_20 = high_series.rolling(20).max().iloc[-1]
        support_20 = low_series.rolling(20).min().iloc[-1]

        breakout_up_dist = (close - resistance_20) / (resistance_20 + 1e-9)
        breakout_down_dist = (close - support_20) / (support_20 + 1e-9)

        # -------------------------
        # ATR
        # -------------------------
        tr = np.maximum(
            high_series - low_series,
            np.maximum(
                abs(high_series - close_series.shift(1)),
                abs(low_series - close_series.shift(1))
            )
        )

        atr_14 = tr.ewm(alpha=1/14).mean().iloc[-1]

        # -------------------------
        # CONFIDENCE
        # -------------------------
        trend_conf = np.tanh(
            5 * abs((ema_20 - ema_50) / (ema_50 + 1e-9))
            + 2 * abs(ema20_slope)
        )

        mean_conf = np.tanh(
            0.5 * abs(z_score_20)
            + 2 * abs(
                0.5 - (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
            )
        )

        breakout_conf = 1 / (1 + np.exp(
            -(8 * (close - resistance_20) / (atr_14 + 1e-9)
            + 1.5 * volume_ratio)
        ))

        return {
            "return_1": return_1,
            "log_return": log_return,
            "vol_7": vol_7,
            "vol_14": vol_14,
            "vol_regime": vol_regime,
            "momentum_7": momentum_7,
            "momentum_14": momentum_14,
            "trend_ratio": trend_ratio,
            "trend_strength": trend_strength,
            "hl_range": hl_range,
            "body": body,
            "volume_ratio": volume_ratio,
            "rsi": rsi,
            "mean_20": mean_20,
            "std_20": std_20,
            "z_score_20": z_score_20,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_diff": ema_diff,
            "ema_ratio": ema_ratio,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "resistance_20": resistance_20,
            "support_20": support_20,
            "breakout_up_dist": breakout_up_dist,
            "breakout_down_dist": breakout_down_dist,
            "trend_conf": trend_conf,
            "mean_conf": mean_conf,
            "breakout_conf": breakout_conf,
            "close": close,
            "atr_14": atr_14
        }
    # =========================
    # RSI
    # =========================
    def _rsi(self, series, period=14):

        delta = series.diff()

        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()

        rs = gain / (loss + 1e-9)

        return 100 - (100 / (1 + rs))