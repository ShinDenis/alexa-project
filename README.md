# Amazon Alexa Reviews — Sentiment Analysis

Проект по классификации отзывов на устройства Amazon Alexa.

Бинарная классификация, определение характера отзыва: **Positive / Negative**.

---
## Основные проблемы
 Дисбаланс выборки осложняет обучение модели.

## Ключевые решения
Акцент сделан на предсказании негативного класса.

- Двойной TF-IDF **NLP:** NLTK (стемминг + лемматизация) с сохранением слов-отрицаний.
- Feature Engineering:
  - обработаны пропуски и дубликаты,
  - добавлены переменные:
    - тип устройства `device_type`
    - длина отзыва`review_length`
- Тюнинг параметров с приоритетом на F1, recall негативного класса 
- Подбор порога принятия решений по F1 негативного класса (0.414 → 0.699)
- Выбрана модель LinearSVC с калибровкой вероятностей через `CalibratedClassifierCV`
---

## Результаты моделей

| Модель | Accuracy | ROC-AUC | Negative F1 |
|---|---|---|---|
| LogisticRegression | 0.933 | 0.950 | 0.682 |
| RandomForest | 0.926 | 0.928 | 0.588 |
| GradientBoosting | 0.946 | 0.950 | 0.689 |
| XGBoost | 0.916 | 0.886 | 0.506 |
| Naive Bayes | 0.938 | 0.921 | 0.536 |
| KNN | 0.919 | 0.792 | 0.490 |
| **LinearSVC** ✅ | **0.938** | **0.949** | **0.699** |
| SuperLearner | 0.941 | — | 0.700 |

**Лучшая модель — LinearSVC** с калиброванными вероятностями и подобранным порогом 0.77.

---

## Структура репозитория

```
amazon-alexa-sentiment/
├── app/
│   ├── app.py
│   ├── model.pkl
│   └── templates/
│       └── index.html
├── notebooks/
│   ├── Amazon_Alexa_analysis.ipynb          # EDA анализ
│   └── Amazon_Alexa_fe_model_pipeline.ipynb # Инженерия, модели
├── dataset/
│   └── amazon_alexa.tsv
├── render.yaml
├── requirements.txt
└── README.md
```


---
## Демо.

