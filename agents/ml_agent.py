def ml_model_predict(last_row_df):
    """
    تأخذ DataFrame يحتوي على سطر واحد (آخر شمعة) وتعطي قرار
    """
    try:
        # تحميل الموديل والـ Scaler
        model = joblib.load(r"C:\project\stock market\code - Copy\model\trained_xgb_model.pkl")
        scaler = joblib.load(r"C:\project\stock market\code - Copy\model\scaler.pkl")
        
        features = [
            'return_1', 'return_3', 'return_5', 'vol_7', 'vol_14',
            'ma_10', 'ma_20', 'trend', 'hl_range', 'vol_change', 'market_return'
        ]
        
        # تجهيز البيانات
        X = last_row_df[features]
        X_scaled = scaler.transform(X)
        
        # التنبؤ بالاحتمالية
        proba = model.predict_proba(X_scaled)[:, 1][0]
        
        # تطبيق منطق الـ Classification Layer الخاص بك
        if proba > 0.6:
            decision = 'BUY'
            confidence = (proba - 0.6) / 0.4
        elif proba < 0.4:
            decision = 'SELL'
            confidence = (0.4 - proba) / 0.4
        else:
            decision = 'HOLD'
            confidence = 1 - (abs(proba - 0.5) / 0.1)
            
        return {
            "decision": decision,
            "probability": float(proba),
            "confidence": float(confidence.clip(0, 1))
        }
    except Exception as e:
        print(f"⚠️ ML Prediction Error: {e}")
        return None