
#  AI Crypto Trading System (TFT + XGBoost + Technical Analysis)

نظام تداول آلي هجين متطور يعتمد على معالجة البيانات الحية (Real-time Streaming) ودمج قرارات ثلاثة نماذج ذكاء اصطناعي وخوارزميات برمجية لضمان أعلى دقة في التنبؤ باتجاه أسعار العملات الرقمية.

## 🌟 المميزات (Features)
*   **بث حي (Real-time Streaming):** اتصال مستمر عبر WebSockets لاستقبال بيانات الأسعار لحظة بلحظة.
*   **نظام هجين (Hybrid Ensemble):** دمج ثلاثة محركات تحليلية:
    1.  **TFT (Temporal Fusion Transformer):** المحرك الرئيسي للتنبؤ بالسلاسل الزمنية.
    2.  **XGBoost Classifier:** نموذج تعلم آلي لتصنيف اتجاه السوق.
    3.  **Technical Strategy:** خوارزمية تعتمد على EMA و Z-Score ومستويات الدعم والمقاومة.
*   **معالجة متوازية (Asynchronous Processing):** استخدام الـ Threading لضمان عدم تأثر البث المباشر بعمليات التحليل الثقيلة.
*   **دمج ذكي للقرار (Weighted Decision):** نظام تصويت مرجح يعطي الأولوية للنموذج الأدق (TFT).

## 🏗 هيكلية النظام (System Architecture)
```text
├── agents/
│   ├── stream_agent.py          # إدارة الاتصال بالـ WebSocket
│   ├── tft_agent.py             # نموذج الـ Temporal Fusion Transformer
│   ├── ml_agent.py              # نموذج الـ XGBoost والـ Scaler
│   ├── strategy_agent.py        # الاستراتيجية الفنية الكلاسيكية
│   └── final_decision_agent.py  # المجمع النهائي للقرار (Weighted Scorer)
├── core/
│   ├── candle_builder.py        # تجميع البيانات وتحويلها لشموع
│   ├── feature_agent.py         # حساب المؤشرات الفنية لحظياً
│   └── create_history.py        # بناء ملف البيانات التاريخي
└── main.py                      # نقطة انطلاق النظام (LangChain Orchestrator)
```

## 🛠 آلية اتخاذ القرار (Decision Logic)
يتم اتخاذ القرار بناءً على معادلة حسابية مرجحة:
*   **TFT Model:** 45% من الوزن الإجمالي.
*   **XGBoost Model:** 35% من الوزن الإجمالي.
*   **Technical Strategy:** 20% من الوزن الإجمالي.




