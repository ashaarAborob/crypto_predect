import numpy as np

def get_final_consensus(ml_res, strat_res, tft_res):
    """
    الـ Agent المسؤول عن دمج النتائج بعملية حسابية مرجحة
    """
    # 1. تحديد أوزان الموديلات (يمكنك تعديلها حسب دقة كل موديل عندك)
    weights = {
        'ml_xgboost': 0.35,  # الموديل اللي تدرب على بيانات ضخمة
        'tft_ai': 0.45,      # موديل التنبؤ الزمني
        'strategy': 0.20     # المؤشرات الفنية الكلاسيكية
    }

    # 2. تحويل القرارات إلى قيم عددية
    # BUY = 1, SELL = -1, HOLD = 0
    mapping = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
    
    scores = []
    
    # حساب قيمة XGBoost
    if ml_res:
        val = mapping.get(ml_res['decision'], 0)
        conf = ml_res.get('confidence', 0.5)
        scores.append(val * conf * weights['ml_xgboost'])
    
    # حساب قيمة Strategy
    if strat_res:
        val = mapping.get(strat_res['action'], 0)
        conf = strat_res.get('confidence', 0.5)
        scores.append(val * conf * weights['strategy'])
        
    # حساب قيمة TFT
    if tft_res:
        val = mapping.get(tft_res['action'], 0)
        conf = tft_res.get('confidence', 0.5)
        scores.append(val * conf * weights['tft_ai'])

    # 3. العملية الحسابية النهائية (Sum of Weighted Scores)
    final_score = sum(scores)
    
    # 4. اتخاذ القرار بناءً على عتبة (Threshold)
    # إذا كان السكور موجب وقوي (أكبر من 0.2 مثلاً) -> BUY
    # إذا كان سالب وقوي (أصغر من -0.2) -> SELL
    threshold = 0.15 
    
    if final_score > threshold:
        decision = "BUY"
    elif final_score < -threshold:
        decision = "SELL"
    else:
        decision = "HOLD"
        
    return {
        "decision": decision,
        "score": round(final_score, 4),
        "details": {
            "ml": ml_res['decision'] if ml_res else "N/A",
            "strat": strat_res['action'] if strat_res else "N/A",
            "tft": tft_res['action'] if tft_res else "N/A"
        }
    }