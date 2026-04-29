import websocket
import json
import time


class StreamAgent:

    def __init__(self, candle_builder, on_candle_close):
        self.cb = candle_builder
        self.on_close = on_candle_close

        self.ws = None

    # -------------------------
    # START WITH INPUT
    # -------------------------
    def run(self, symbol):
        self.symbol = symbol.upper()

        socket = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"

        self.ws = websocket.WebSocketApp(
            socket,
            on_message=self.observe,
            on_error=self.on_error,
            on_close=self.on_close_ws,
            on_open=self.on_open
        )

        self.ws.run_forever()

    # -------------------------
    # STREAM HANDLER
    # -------------------------
    def observe(self, ws, message):
        try:
            data = json.loads(message)

            price = float(data["p"])
            timestamp = data["T"]
            symbol = data["s"]
            quantity = float(data["q"])

            candle = self.cb.on_tick(symbol, price, quantity, timestamp)

            if candle:
                self.on_close(symbol, candle)

        except Exception as e:
            print("⚠️ error:", e)

    # -------------------------
    # BASIC EVENTS
    # -------------------------
    def on_error(self, ws, error):
        print("❌ error:", error)

    def on_close_ws(self, ws, *args):
        print("⚠️ closed")

    def on_open(self, ws):
        print("✅ connected:", self.symbol)