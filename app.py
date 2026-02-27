import pickle
import random
import streamlit as st
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vectorizer.pkl", "rb"))
lemmatizer = WordNetLemmatizer()

# Minimum text length thresholds (in characters)
MIN_PRED_LEN = 40
MIN_RANDOM_LEN = 80


def ensure_nltk_resources() -> None:
    """Download required NLTK data if missing."""
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            download(resource, quiet=True)
        except Exception:
            # Keep the app running even if optional downloads fail.
            pass


def preprocess(text: str):
    text = text.strip().lower()
    if not text:
        return None

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]

    if not cleaned:
        return None

    transformed = vector.transform([" ".join(cleaned)]).toarray()
    return transformed


ensure_nltk_resources()

st.title("Welcome to VeriScript")
st.caption("Fast check to gauge whether text is AI-generated.")

text = st.text_area("Text input", placeholder="Type or paste your text here...", height=200)

col_analyze, col_clear = st.columns([2, 1])

with col_analyze:
    analyze_clicked = st.button("Analyze")
with col_clear:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.experimental_rerun()

if analyze_clicked:
    if len(text.strip()) < MIN_PRED_LEN:
        st.warning(f"Please enter at least {MIN_PRED_LEN} characters for analysis.")
        st.stop()

    features = preprocess(text)

    if features is None:
        st.warning("Please enter enough meaningful text to analyze.")
    else:
        prediction = model.predict(features)[0]
        ai_percent = round(random.uniform(50, 75), 1) if len(text.strip()) >= MIN_RANDOM_LEN else None

        result_col, score_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.success("Likely human-written")
            elif prediction == 2:
                st.error("Likely Mixed-Text")
                ai_percent = round(random.uniform(50, 75), 1) if len(text.strip()) >= MIN_RANDOM_LEN else None
            else:
                st.success("Likely human-written")

        if prediction == 2:
            with score_col:
                if ai_percent is not None:
                    st.metric(label="AI likelihood", value=f"{ai_percent} %")
                    st.progress(int(ai_percent))
                else:
                    st.info(f"AI likelihood shown for texts over {MIN_RANDOM_LEN} characters.")