# ✈️ Twitter Airline Sentiment Analysis — ML Pipeline

End-to-end NLP pipeline for 3-class sentiment classification on real airline tweets. Includes text preprocessing, TF-IDF feature extraction, 3 ML models, statistical validation, Prophet time-series forecasting, and a live Streamlit app.

---

## 📊 Dataset

**Twitter US Airline Sentiment** — 14,640 real tweets, February 2015

| Attribute | Value |
|---|---|
| Records | 14,640 real tweets |
| Classes | Negative (63%), Neutral (21%), Positive (16%) |
| Airlines | United, US Airways, American, Southwest, Delta, Virgin America |
| Timestamps | Real — Feb 17–24, 2015 (181 hourly points for forecasting) |
| Total Columns | 25 (15 original + 10 engineered) |

> Majority class baseline (always predict Negative): **62.7%**

---

## 📁 Project Structure

```
airline-sentiment-ml-pipeline/
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Distributions, airline breakdown, volume over time
│   ├── 02_preprocessing.ipynb           # URL removal, @mention removal, stopwords
│   ├── 03_feature_engineering.ipynb     # TF-IDF, train/test split, SMOTE, top terms
│   ├── 04_model_training.ipynb          # LR, RF, SVM — training, CM, comparison
│   ├── 05_evaluation.ipynb              # CV summary, Wilcoxon, Mann-Whitney, per-class
│   └── 06_forecasting.ipynb             # Prophet hourly forecast, 3 scenarios, decomposition
│
├── data/
│   └── Tweets.csv                       # Source dataset
│
├── results/                             # Saved plots
│
├── app/
│   └── app.py                           # Streamlit real-time prediction app
│
└── requirements.txt
```

---

## 📈 Results

| Model | CV F1 (5-fold) | Test Accuracy | Test F1 (weighted) | vs Baseline |
|---|---|---|---|---|
| Logistic Regression | 0.839 | 0.769 | 0.773 | +14.6% |
| Random Forest | 0.847 | 0.762 | 0.757 | +13.0% |
| SVM (RBF) | 0.851 | 0.764 | 0.762 | +13.5% |

**Majority class baseline: 62.7%** — all models beat this by >13% on real text features.

---

## 🔄 Pipeline

```
Raw Tweet → Preprocess → TF-IDF (5000 features, bigrams) → Split (80/20) → SMOTE → Train → Evaluate → Forecast
```

**Why each step:**
- **Stopword removal + URL/mention stripping** — removes noise specific to tweets
- **Bigrams** — captures sentiment phrases like "cancelled flight", "great service"
- **SMOTE on train only** — corrects 63/21/16% imbalance without leaking test data
- **5-fold CV** — more reliable than single train/test split
- **Wilcoxon + Mann-Whitney** — statistically validates performance differences between models
- **Prophet forecasting** — 181 hourly points, sentiment as external regressor, 3 future scenarios

---

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

---

## 🛠️ Tech Stack

| Area | Library |
|---|---|
| NLP | scikit-learn TfidfVectorizer |
| ML | scikit-learn (LR, RF, SVM) |
| Imbalance | imbalanced-learn (SMOTE) |
| Statistics | scipy (Wilcoxon, Mann-Whitney U) |
| Forecasting | Prophet |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit |

---

## ⚠️ Limitations

- Dataset covers only 8 days — forecasting captures short-term patterns only
- Domain-specific (airline complaints) — pipeline needs retraining for other domains

---

*Made by [Prathicksha Selvaraj](https://github.com/prathickshaselvaraj)*
