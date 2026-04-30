import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def tft_model():

    csv_PATH        = r"C:\project\stock market\code\core\history.csv"
    CHECKPOINT_PATH = r"C:\project\stock market\code\model\last-v10.ckpt"
    HORIZON         = 7
    ENCODER_LENGTH  = 30
    BUY_THRESHOLD   = 0.0015
    SELL_THRESHOLD  = -0.0015

    # ─────────────────────────────
    # 1. LOAD
    # ─────────────────────────────
    df = pd.read_csv(csv_PATH)

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("min").dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)

    # ─────────────────────────────
    # 2. VALIDATION
    # ─────────────────────────────
    required = [
        "log_return","volatility_7","volatility_14","volatility_30",
        "vol_regime","return_mean_7","return_mean_14",
        "momentum_7","momentum_14","trend_strength",
        "rsi_14","rsi_momentum",
        "close_lag_1","close_lag_2","close_lag_3","close_lag_7","close_lag_14",
        "return_lag_1","return_lag_2","return_lag_3","return_lag_7","return_lag_14",
        "lag_momentum_1",
        "volume_change","volume_ma_7","volume_ratio",
        "buy_pressure","sell_pressure",
        "high_low_range","close_open_range","volatility_squeeze",
        "day_of_week","month","week_of_year","is_month_end",
        "coin_name","time_idx","target",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) < ENCODER_LENGTH + HORIZON:
        raise ValueError("Not enough data")

    # ─────────────────────────────
    # 3. FEATURES LIST
    # ─────────────────────────────
    KNOWN_REALS = ["time_idx","day_of_week","month","week_of_year","is_month_end"]

    UNKNOWN_REALS = [c for c in required if c not in KNOWN_REALS + ["coin_name","target"]]

    # ─────────────────────────────
    # 4. DATASET
    # ─────────────────────────────
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["coin_name"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=HORIZON,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        static_categoricals=["coin_name"],
        target_normalizer=GroupNormalizer(groups=["coin_name"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # ─────────────────────────────
    # 5. MODEL
    # ─────────────────────────────
    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()

    # ─────────────────────────────
    # 6. PREDICT
    # ─────────────────────────────
    predictions = model.predict(loader, mode="quantiles")

    q10 = predictions[:, :, 0]
    q50 = predictions[:, :, 1]
    q90 = predictions[:, :, 2]

    # ─────────────────────────────
    # 7. SIGNAL
    # ─────────────────────────────
    last_pred = q50[0, -1].item()

    if last_pred > BUY_THRESHOLD:
        action = "BUY"
    elif last_pred < SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    buy_prob  = float((q50[0] > BUY_THRESHOLD).mean())
    sell_prob = float((q50[0] < SELL_THRESHOLD).mean())
    hold_prob = 1 - (buy_prob + sell_prob)

    confidence = float(max(buy_prob, sell_prob, hold_prob))

    predictions_per_day = []
    for day in range(HORIZON):
        predictions_per_day.append({
            "day": day + 1,
            "q10": float(q10[0, day].item()),
            "q50": float(q50[0, day].item()),
            "q90": float(q90[0, day].item()),
        })

    return {
        "action": action,
        "confidence": confidence,
        "probs": {
            "BUY": buy_prob,
            "SELL": sell_prob,
            "HOLD": hold_prob
        },
        "predictions": predictions_per_day
    }