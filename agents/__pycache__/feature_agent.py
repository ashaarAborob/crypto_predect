import pandas as pd
import requests 
from datetime import datetime, timedelta
class FeatureAgent:

    def __init__(self, history_path, symbol, window_days=30):
        self.history_path = history_path
        self.symbol = symbol
        self.window = window_days

        # load existing if exists
        try:
            self.df = pd.read_excel(history_path)
        except:
            self.df = pd.DataFrame()

    # -------------------------
    # BOOTSTRAP HISTORY
    # -------------------------
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

        rows = []

        for d in data:
            rows.append({
                "timestamp": d[0],
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5])
            })

        self.df = pd.DataFrame(rows)

        self.save()

        print("✅ history loaded:", len(self.df))

    # -------------------------
    # SAVE
    # -------------------------
    def save(self):
        self.df.to_parquet(self.history_path, index=False)

    # -------------------------
    # UPDATE WITH STREAM
    # -------------------------
    def update(self, candle):

        self.df = pd.concat([
            self.df,
            pd.DataFrame([candle])
        ], ignore_index=True)

        # keep last 30 days
        self.df = self.df.tail(self.window)

        self.save()

        return self.df