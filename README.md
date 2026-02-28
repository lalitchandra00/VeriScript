# VeriScript

Fast, lightweight Streamlit app to gauge whether pasted text is AI-generated, human-written, or mixed. It loads a pre-trained classifier (`model.pkl`) and vectorizer (`vectorizer.pkl`), cleans text with NLTK, and surfaces a simple verdict with an optional AI-likelihood meter.

## Features
- One-click analysis via Streamlit with clear human / mixed / AI outcomes
- Basic text cleaning with tokenization, stopword removal, and lemmatization
- Guardrails for too-short or low-signal inputs
- Optional AI-likelihood indicator for longer passages

## Tech Stack
- Python, Streamlit UI
- NLTK for preprocessing (stopwords, tokenization, lemmatization)
- scikit-learn vectorizer + pickled classification model


## How It Works (high level)
1) Clean input: lowercase, tokenize, remove stopwords, keep alphabetic tokens, lemmatize
2) Vectorize: transform cleaned text with the fitted vectorizer
3) Predict: load the pickled model and classify into human / mixed / AI
4) Display: show verdict; for longer text, surface an AI-likelihood meter

## Project Structure
- `app.py` — Streamlit UI and inference pipeline
- `requirements.txt` — Python dependencies
- `model.pkl` / `vectorizer.pkl` — required artifacts (not tracked)
- `model.ipynb` — optional notebook for experimentation/training

## Troubleshooting
- Missing NLTK data: run the downloader command above.
- Model compatibility errors: ensure `model.pkl` and `vectorizer.pkl` were trained with the same scikit-learn version listed in requirements.
- UI not loading: confirm Streamlit is installed and run from the project root.

## Next Steps
- Document the training workflow in `model.ipynb`
- Add evaluation metrics and example inputs/outputs
- Optionally expose a simple REST endpoint (FastAPI) alongside the Streamlit UI
