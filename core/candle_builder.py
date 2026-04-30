class CandleBuilder:
    def __init__(self, timeframe_seconds=3):
        self.tf = timeframe_seconds * 1000
        self.candles = {}

    def on_tick(self, pair, price, quantity, timestamp):

        if pair not in self.candles:
            self.candles[pair] = self._new(price, quantity, timestamp)

        c = self.candles[pair]

        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
        c["volume"] += quantity   # 🔥 جمع volume الحقيقي

        if (timestamp - c["timestamp"]) >= self.tf:
            finished = self._format(pair, c)
            self.candles[pair] = self._new(price, quantity, timestamp)
            return finished

        return None

    def _new(self, price, quantity, timestamp):
        return {
            "timestamp": timestamp,   # ✅ بدل start
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": quantity        # ✅ بداية volume
        }

    def _format(self, pair, c):
        return {
            "symbol": pair,
            "timestamp": c["timestamp"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"]
        }
    

