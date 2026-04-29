from pytorch_forecasting import TemporalFusionTransformer
import numpy as np
import pandas as pd
import torch



model = TemporalFusionTransformer.load_from_checkpoint("C:\\project\\stock market\\code\\model\\tft_pipeline\\best_model.ckpt")
dataset="C:\\project\\stock market\\code\\model\\tft_pipeline\\dataset.pkl"
import numpy as np
import pandas as pd
import torch

def predict_from_features(
    features: dict,
    model,
    training_dataset,
    coin_name="unknown",
    history_df=None
):
    """
    features: output من FeatureAgent (dict)
    model: TemporalFusionTransformer
    training_dataset: dataset.pkl
    history_df: آخر تاريخ (مهم لتكوين sequence)
    """

    # =========================
    # 1. تحويل features إلى DataFrame
    # =========================
    feat_df = pd.DataFrame([features])

    # إضافة معلومات أساسية
    feat_df["coin_name"] = coin_name
    feat_df["time_idx"] = 0

    # =========================
    # 2. دمج مع history (مهم جداً)
    # =========================
    if history_df is not None:
        df = pd.concat([history_df, feat_df], ignore_index=True)
    else:
        df = feat_df

    # =========================
    # 3. بناء dataset للموديل
    # =========================
    dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        predict=True,
        stop_randomization=True
    )

    loader = dataset.to_dataloader(train=False, batch_size=1)

    # =========================
    # 4. prediction
    # =========================
    pred = model.predict(loader, mode="quantiles")

    q10 = pred[:, :, 0].cpu().numpy().flatten()
    q50 = pred[:, :, 1].cpu().numpy().flatten()
    q90 = pred[:, :, 2].cpu().numpy().flatten()

    # =========================
    # 5. SMART DECISION LAYER
    # =========================

    trend = np.mean(q50)
    confidence = np.mean(q90 - q10)
    volatility = np.std(q50)

    # thresholds (تقدر تعدليهم)
    TREND_TH = 0.0015
    CONF_TH = 0.0010
    VOL_TH = 0.0020

    # filters
    if confidence < CONF_TH:
        return "NO TRADE ❌ low confidence"

    if volatility > VOL_TH:
        return "NO TRADE ❌ unstable market"

    # decision
    if trend > TREND_TH:
        return "BUY 🟢"

    elif trend < -TREND_TH:
        return "SELL 🔴"

    else:
        return "HOLD 🟡"
    

# =========================================================
# 🧠 SMART DECISION LAYER
# =========================================================
def smart_decision(pred):

    q10 = pred[:, :, 0].cpu().numpy().flatten()
    q50 = pred[:, :, 1].cpu().numpy().flatten()
    q90 = pred[:, :, 2].cpu().numpy().flatten()

    # -------------------------
    # TREND (7 days)
    # -------------------------
    trend = np.mean(q50)

    # -------------------------
    # CONFIDENCE
    # -------------------------
    confidence = np.mean(q90 - q10)

    # -------------------------
    # STABILITY
    # -------------------------
    volatility = np.std(q50)

    # -------------------------
    # THRESHOLDS
    # -------------------------
    TREND_TH = 0.0015
    CONF_TH  = 0.0010
    VOL_TH   = 0.0020

    # -------------------------
    # FILTERS FIRST
    # -------------------------
    if confidence < CONF_TH:
        return "NO TRADE ❌ (low confidence)"

    if volatility > VOL_TH:
        return "NO TRADE ❌ (unstable market)"

    # -------------------------
    # FINAL DECISION
    # -------------------------
    if trend > TREND_TH:
        return "BUY 🟢"

    elif trend < -TREND_TH:
        return "SELL 🔴"

    else:
        return "HOLD 🟡"


signal = predict_from_features(
    features=feature_dict,
    model=model,
    training_dataset=dataset,
    coin_name="BTCUSDT",
    history_df=history
)

print(signal)