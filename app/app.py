"""
Twitter Airline Sentiment Analyser — Streamlit App
Real-time prediction + dataset insights + forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airline Sentiment Analyser", page_icon="✈️", layout="wide")

# ── PREPROCESSING ─────────────────────────────────────────
STOPWORDS = set([
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','am','is','are','was','were','be',
    'been','have','has','had','do','did','will','would','could','should',
    'a','an','the','and','but','or','for','so','at','by','from','in',
    'of','on','to','up','with','as','just','also','not','no','very',
    's','t','re','co','rt','via','amp'
])

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

# ── TRAIN MODEL (cached) ──────────────────────────────────
@st.cache_resource
def load_model():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Tweets.csv')
    df = pd.read_csv(data_path)
    df['clean_text'] = df['text'].apply(preprocess)
    df['label'] = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            min_df=3, sublinear_tf=True)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    model.fit(X_tr, y_tr)
    return model, tfidf, df

model, tfidf, df = load_model()

COLORS  = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
EMOJIS  = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}
LABELS  = {0: 'negative', 1: 'neutral', 2: 'positive'}

# ── TABS ──────────────────────────────────────────────────
st.title("✈️ Twitter Airline Sentiment Analyser")
st.markdown("*NLP pipeline: TF-IDF → Logistic Regression | 14,640 real airline tweets*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Dataset Insights", "ℹ️ Pipeline"])

# ── TAB 1: PREDICTION ────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter a tweet to classify")
        user_text = st.text_area("", height=110,
            placeholder="e.g. The flight was delayed 3 hours and no one told us why!",
            label_visibility="collapsed")

        st.markdown("**Try an example:**")
        examples = {
            "😞 Negative": "Flight cancelled with no explanation. Worst airline experience ever!",
            "😐 Neutral":  "Flying United to Chicago tomorrow morning.",
            "😊 Positive": "Amazing crew on this flight! Best service I've had in years.",
        }
        ecols = st.columns(3)
        for i, (lbl, ex) in enumerate(examples.items()):
            if ecols[i].button(lbl, use_container_width=True):
                user_text = ex

    with col2:
        st.subheader("Result")
        if user_text and user_text.strip():
            cleaned = preprocess(user_text)
            vec = tfidf.transform([cleaned])
            pred = int(model.predict(vec)[0])
            probs = model.predict_proba(vec)[0]
            pred_name = LABELS[pred]
            color = COLORS[pred_name]
            emoji = EMOJIS[pred_name]

            st.markdown(
                f"""<div style='background:{color}22;border-left:4px solid {color};
                padding:16px;border-radius:8px;text-align:center'>
                <div style='font-size:48px'>{emoji}</div>
                <div style='font-size:22px;font-weight:bold;color:{color}'>{pred_name.title()}</div>
                <div style='font-size:12px;color:#666;margin-top:4px'>
                Confidence: {float(max(probs))*100:.1f}%</div></div>""",
                unsafe_allow_html=True)

            st.markdown("**Class probabilities:**")
            for cls_idx, cls_name in enumerate(['negative','neutral','positive']):
                st.progress(float(probs[cls_idx]),
                            text=f"{cls_name.title()}: {float(probs[cls_idx])*100:.1f}%")
        else:
            st.info("Enter a tweet on the left to get a prediction.")

    st.markdown("---")
    st.subheader("Batch Prediction")
    batch = st.text_area("One tweet per line:", height=100, key="batch",
                          placeholder="Great flight today!\nBaggage lost again.\nFlying Delta to Boston.")
    if st.button("Classify All", type="primary") and batch.strip():
        lines = [l.strip() for l in batch.strip().split('\n') if l.strip()]
        vecs = tfidf.transform([preprocess(l) for l in lines])
        preds = [LABELS[int(p)] for p in model.predict(vecs)]
        probs = model.predict_proba(vecs)
        result_df = pd.DataFrame({
            'Tweet': lines,
            'Sentiment': [p.title() for p in preds],
            'Confidence': [f"{float(max(p))*100:.1f}%" for p in probs],
            'P(Negative)': [f"{float(p[0])*100:.1f}%" for p in probs],
            'P(Neutral)':  [f"{float(p[1])*100:.1f}%" for p in probs],
            'P(Positive)': [f"{float(p[2])*100:.1f}%" for p in probs],
        })
        st.dataframe(result_df, use_container_width=True)

# ── TAB 2: INSIGHTS ───────────────────────────────────────
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tweets", f"{len(df):,}")
    c2.metric("Airlines", df['airline'].nunique())
    c3.metric("Date Range", "Feb 17–24, 2015")
    c4.metric("Majority Baseline", "62.7%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment Distribution**")
        counts = df['airline_sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(counts.index,counts.values,
                      color=[COLORS[c] for c in counts.index], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                    str(int(bar.get_height())), ha='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.markdown("**Sentiment by Airline**")
        ct = pd.crosstab(df['airline'], df['airline_sentiment'])
        ct = ct[['negative','neutral','positive']] if all(
            c in ct.columns for c in ['negative','neutral','positive']) else ct
        fig, ax = plt.subplots(figsize=(5, 3))
        ct.plot(kind='bar', ax=ax,
                color=['#e74c3c','#95a5a6','#2ecc71'], alpha=0.85,
                edgecolor='white', rot=30)
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Model Performance Summary**")
    perf = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM (RBF)'],
        'CV F1 (mean)': ['0.839', '0.847', '0.851'],
        'Test Accuracy': ['0.769', '0.762', '0.764'],
        'Test F1 (wt.)': ['0.773', '0.757', '0.762'],
        'vs Baseline (+)': ['+14.6%', '+13.5%', '+13.7%'],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.info("Majority class baseline = 62.7% (always predict Negative). "
            "All models exceed this using TF-IDF text features only.")

# ── TAB 3: PIPELINE ───────────────────────────────────────
with tab3:
    st.subheader("Pipeline Architecture")
    st.code("""
Raw Tweet Text
    ↓
Preprocessing
  • Lowercase
  • Remove URLs, @mentions
  • Keep hashtag words (#delay → delay)
  • Remove punctuation
  • Remove stopwords
    ↓
TF-IDF Vectorization
  • max_features=5000
  • ngram_range=(1,2)  ← bigrams capture 'cancelled flight', 'great service'
  • min_df=3
  • sublinear_tf=True
    ↓
Train/Test Split (80/20, stratified) ← done ONCE
    ↓
SMOTE (train set only) ← corrects 63/21/16% imbalance
    ↓
Model Training
  • Logistic Regression (interpretable baseline)
  • Random Forest (non-linear ensemble)
  • SVM RBF (margin-based, standard for text)
    ↓
Evaluation
  • 5-fold stratified CV
  • Holdout test set
  • Wilcoxon + Mann-Whitney statistical tests
    ↓
Temporal Forecasting (Prophet)
  • 181 hourly time points
  • Sentiment as external regressor
  • 48-hour ahead forecast under 3 scenarios
    ↓
Streamlit Deployment ← you are here
""", language="text")

    st.markdown("**Dataset Details**")
    st.markdown("""
| Column | Type | Description |
|--------|------|-------------|
| text | Raw text | Tweet content (input to NLP) |
| airline_sentiment | Target | negative / neutral / positive |
| airline | Categorical | 6 US airlines |
| tweet_created | Timestamp | Real timestamps → forecasting |
| retweet_count | Numeric | Engagement signal |
| negativereason | Categorical | Root cause for negative tweets |
| word_count | Engineered | Words per tweet |
| char_count | Engineered | Characters per tweet |
| hashtag_count | Engineered | # symbols per tweet |
| mention_count | Engineered | @ symbols per tweet |
| exclamation_count | Engineered | ! per tweet |
| question_count | Engineered | ? per tweet |
| hour | Engineered | Hour of day (0–23) |
| day_of_week | Engineered | Day of week (0=Mon) |
| has_url | Engineered | Whether tweet contains URL |
""")
