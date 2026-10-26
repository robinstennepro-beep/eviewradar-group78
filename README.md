# ReviewRadar

**Guest reviews in, prioritised action plan and revenue business case out.**

ReviewRadar reads hotel guest reviews, finds what guests love and hate across 8 hospitality aspects, measures how
many stars each complaint really costs, and tells the General Manager what to fix first and what it is worth in
RevPAR. A second view benchmarks guest sentiment across cities for investors and asset managers.

Group 78 · Advanced AI Algorithms · ESSEC MSc IMHI

## Use it

Open the hosted app: **https://reviewradar-group78.streamlit.app** (Streamlit Community Cloud). If the page says the app is waking up, wait about 30 seconds.

1. **Analyse a hotel**: pick a demo data set, upload a CSV (one review per row, optional rating column) or paste
   reviews, then click *Run ReviewRadar*.
2. **Market benchmark**: compare guest sentiment across Paris, London, Amsterdam, Barcelona, New York and Berlin.
3. **Demo video**: the 4:40 product demo.
4. **Method & sources**.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What is in this repository

| File | Role |
|---|---|
| `streamlit_app.py` | the web app (interface) |
| `reviewradar_core.py` | the AI engine: opinion mining, aspect sentiment, driver model, charts, action plan |
| `requirements.txt` | Python packages installed by Streamlit Cloud |
| `.streamlit/config.toml` | colours of the app |
| `data/tripadvisor_hotel_reviews.csv` | TripAdvisor Hotel Reviews, 20,491 reviews (Kaggle) |
| `data/upload_example_40_reviews.csv` | small file to test the upload |
| `assets/ReviewRadar_demo_Group78.mp4` | demo video |

The hosted version runs on a free CPU server, so it uses the **lite engine** (VADER sentiment lexicon and a
TF-IDF + logistic regression star model trained on the Kaggle data). The Hugging Face transformer engine
(DeBERTa-v3 aspect-based sentiment) and the optional LLM memo run in the Colab notebook of the full project package.

## Data and credits

TripAdvisor Hotel Reviews, Kaggle dataset by Larxel; original source Alam, Ryu & Lee (2016), *Information
Sciences*, 339, 206-223. Used for an academic project; check the licence on the Kaggle page before any commercial use.
Business case based on Anderson (2012), Cornell Center for Hospitality Research. Importance-Performance Analysis:
Martilla & James (1977).
