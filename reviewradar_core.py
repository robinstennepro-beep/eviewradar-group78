"""
ReviewRadar core engine
=======================

Turns raw hotel guest reviews into:
  1. aspect-level sentiment (Room, Cleanliness, Staff, Location, F&B, Noise, Value, Amenities)
  2. a driver analysis (which aspects actually move the star rating)
  3. an Importance-Performance map ("fix first" vs "keep up the good work")
  4. a prioritised action plan with guest evidence
  5. an illustrative revenue upside (RevPAR) based on published hospitality research

Two interchangeable sentiment engines are available:
  * "hf"   : Hugging Face transformers (DeBERTa-v3 aspect-based sentiment + multilingual BERT stars).
             Best quality. Recommended on a GPU (Google Colab T4 or a Hugging Face Space).
  * "lite" : VADER lexicon + a TF-IDF / logistic-regression star model trained on the Kaggle data.
             Runs anywhere on CPU in seconds, no model download. Used as a fallback.

Everything is deterministic (fixed seeds), so results are reproducible.
"""

from __future__ import annotations

import os
import re
import math
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "tripadvisor_hotel_reviews.csv")
STAR_MODEL_PATH = os.path.join(HERE, "models", "star_model_tfidf_logreg.joblib")

# Hugging Face models used by the "hf" engine (all public, no token required)
HF_ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"            # aspect-based sentiment (MIT)
HF_STARS_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # 1-5 stars, 6 languages (MIT)
HF_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"                     # optional manager memo (Apache-2.0)

# --------------------------------------------------------------------------------------
# 1. Hospitality aspect taxonomy
# --------------------------------------------------------------------------------------
ASPECTS: dict[str, dict] = {
    "Room & Bed": {
        "keywords": ["room", "bed", "bedroom", "pillow", "mattress", "suite", "bathroom", "shower",
                     "tub", "bathtub", "towel", "closet", "balcony", "decor", "furniture", "carpet",
                     "sheets", "linen", "toilet"],
        "actions": [
            "Audit the worst-rated room categories (mattresses, pillows, bathroom fittings) and plan a soft-goods refresh.",
            "Run weekly preventive-maintenance rounds so small defects are fixed before guests report them.",
            "Make room descriptions on OTAs accurate (size, layout) so expectations match reality.",
        ],
    },
    "Cleanliness": {
        "keywords": ["clean", "cleanliness", "dirty", "dust", "dusty", "stain", "stained", "smell",
                     "smelly", "odor", "odour", "mold", "mould", "filthy", "spotless", "hygiene",
                     "bugs", "bedbugs", "cockroach", "cockroaches", "housekeeping", "maid", "hair"],
        "actions": [
            "Introduce supervisor inspection with a signed checklist on a fixed share of rooms every day.",
            "Schedule a deep-clean cycle (bathrooms, carpets, air vents) and odour control.",
            "Align housekeeping staffing with the occupancy forecast from revenue management.",
        ],
    },
    "Staff & Service": {
        "keywords": ["staff", "service", "reception", "receptionist", "front desk", "desk",
                     "concierge", "manager", "management", "employee", "doorman", "bellman",
                     "check-in", "check in", "checkin", "checkout", "check out", "friendly", "rude",
                     "helpful", "unhelpful", "welcoming", "polite", "attentive", "professional"],
        "actions": [
            "Set a check-in time standard and offer online pre-check-in to cut queues at peak hours.",
            "Empower front-desk staff with a service-recovery budget to fix issues on the spot.",
            "Reward staff named positively in reviews; use complaints as weekly training cases.",
        ],
    },
    "Location & View": {
        "keywords": ["location", "located", "walk", "walking", "metro", "subway", "station", "tube",
                     "train", "airport", "beach", "neighborhood", "neighbourhood", "area", "view",
                     "distance", "central", "downtown", "shopping", "transport"],
        "actions": [
            "Publish precise access information (transport, walking times) in listings and confirmation emails.",
            "Create a concierge 'local guide' that turns the neighbourhood into a selling point.",
            "Define room-allocation rules and a 'view guaranteed' rate to monetise and protect views.",
        ],
    },
    "Food & Beverage": {
        "keywords": ["breakfast", "restaurant", "food", "buffet", "dinner", "lunch", "meal", "bar",
                     "coffee", "drink", "cocktail", "wine", "menu", "chef", "room service", "pastries"],
        "actions": [
            "Audit breakfast variety and replenishment at peak time (8 to 10 am).",
            "Review price-to-value of bar and restaurant items versus nearby competitors.",
            "If F&B is not a core strength, partner with local venues rather than under-deliver.",
        ],
    },
    "Noise & Sleep": {
        "keywords": ["noise", "noisy", "quiet", "loud", "soundproof", "soundproofing", "sleep",
                     "slept", "thin walls", "construction", "traffic", "music", "earplugs"],
        "actions": [
            "Prioritise soundproofing (door seals, double glazing) on street-facing rooms.",
            "Allocate light sleepers to quiet floors and offer a sleep kit (earplugs, eye mask).",
            "Enforce quiet hours and schedule works outside sleeping hours.",
        ],
    },
    "Value & Price": {
        "keywords": ["price", "priced", "expensive", "overpriced", "value", "worth", "cheap",
                     "affordable", "money", "cost", "fee", "charge", "charged", "deal", "rate",
                     "bill", "resort fee", "pricey"],
        "actions": [
            "Make all fees transparent at booking (no surprise parking, resort or city fees).",
            "Rebalance pricing in shoulder periods with revenue management; avoid over-pricing low-rated room types.",
            "Bundle value-adds (breakfast, late check-out) rather than discounting the room rate.",
        ],
    },
    "Amenities & Wi-Fi": {
        "keywords": ["wifi", "wi-fi", "internet", "pool", "gym", "fitness", "spa", "sauna",
                     "elevator", "elevators", "lift", "air conditioning", "air conditioner", "ac",
                     "a/c", "tv", "television", "parking", "minibar", "fridge", "kettle",
                     "facilities", "amenities", "safe"],
        "actions": [
            "Upgrade Wi-Fi bandwidth and make a fast tier free for all guests.",
            "Set a maintenance service level for air conditioning, elevators and TVs.",
            "Communicate available amenities before arrival to drive usage and satisfaction.",
        ],
    },
}
ASPECT_NAMES = list(ASPECTS.keys())


def _compile_aspect_patterns() -> dict[str, re.Pattern]:
    pats = {}
    for name, spec in ASPECTS.items():
        kws = sorted(spec["keywords"], key=len, reverse=True)
        alt = "|".join(re.escape(k) for k in kws)
        pats[name] = re.compile(r"\b(" + alt + r")(?:s|es)?\b", flags=re.IGNORECASE)
    return pats


ASPECT_PATTERNS = _compile_aspect_patterns()

_CLAUSE_SPLIT = re.compile(r"[.!?;,\n]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Split a review into short opinion clauses (one idea each)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip() for p in parts if p and len(p.split()) >= 2]


WINDOW = 7  # words kept on each side of an aspect term when a clause is long


def _window(clause: str, start: int, end: int, size: int = WINDOW) -> str:
    """Keep the opinion close to the aspect term (long run-on clauses dilute sentiment)."""
    before = clause[:start].split()
    after = clause[end:].split()
    if len(before) + len(after) <= 2 * size:
        return clause
    return " ".join(before[-size:] + [clause[start:end]] + after[:size]).strip()


def extract_mentions(reviews: pd.Series) -> pd.DataFrame:
    """Return one row per (review, opinion snippet, aspect) with the matched aspect term."""
    rows = []
    for rid, text in reviews.items():
        for clause in split_clauses(text):
            for aspect, pat in ASPECT_PATTERNS.items():
                m = pat.search(clause)
                if m:
                    rows.append({"review_id": rid, "clause": _window(clause, m.start(), m.end()),
                                 "aspect": aspect, "term": m.group(1).lower()})
    return pd.DataFrame(rows, columns=["review_id", "clause", "aspect", "term"])


# --------------------------------------------------------------------------------------
# 2. Sentiment engines
# --------------------------------------------------------------------------------------
class LiteEngine:
    """CPU fallback: VADER lexicon for clauses + TF-IDF/LogReg for stars."""

    name = "lite"
    label = "Lite engine (VADER lexicon + TF-IDF model, CPU)"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        self._star_model = None

    def score_pairs(self, clauses: list[str], terms: list[str], progress=None) -> np.ndarray:
        out = np.zeros(len(clauses))
        for i, c in enumerate(clauses):
            out[i] = self.vader.polarity_scores(c)["compound"]
            if progress and i % 500 == 0:
                progress(i / max(1, len(clauses)), desc="Scoring aspect sentiment")
        return out

    def predict_stars(self, texts: list[str]) -> np.ndarray:
        if self._star_model is None:
            self._star_model = load_or_train_star_model()
        return self._star_model.predict(list(texts)).astype(float)


class HFEngine:
    """Hugging Face transformers engine (recommended on GPU)."""

    name = "hf"
    label = "Hugging Face engine (DeBERTa-v3 ABSA + multilingual BERT)"

    def __init__(self, absa_model: str = HF_ABSA_MODEL, stars_model: str = HF_STARS_MODEL,
                 batch_size: int = 64):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size if self.device == "cuda" else min(batch_size, 16)
        self.absa_tok = AutoTokenizer.from_pretrained(absa_model)
        self.absa = AutoModelForSequenceClassification.from_pretrained(absa_model).to(self.device).eval()
        self._stars_name = stars_model
        self.stars_tok = None
        self.stars = None
        # map label names -> polarity sign, robust to label casing
        id2label = {int(k): str(v).lower() for k, v in self.absa.config.id2label.items()}
        self._pos_idx = [i for i, l in id2label.items() if l.startswith("pos")]
        self._neg_idx = [i for i, l in id2label.items() if l.startswith("neg")]
        if not self._pos_idx or not self._neg_idx:
            raise ValueError(f"Unexpected ABSA labels: {id2label}")

    def score_pairs(self, clauses: list[str], terms: list[str], progress=None) -> np.ndarray:
        """Polarity in [-1, 1] = P(positive) - P(negative) for each (clause, aspect term)."""
        torch = self.torch
        out = []
        n = len(clauses)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                c = clauses[start:start + self.batch_size]
                t = terms[start:start + self.batch_size]
                enc = self.absa_tok(c, t, padding=True, truncation=True, max_length=128,
                                    return_tensors="pt").to(self.device)
                probs = torch.softmax(self.absa(**enc).logits, dim=-1).cpu().numpy()
                out.append(probs[:, self._pos_idx].sum(1) - probs[:, self._neg_idx].sum(1))
                if progress:
                    progress(min(1.0, (start + len(c)) / max(1, n)), desc="Scoring aspect sentiment")
        return np.concatenate(out) if out else np.zeros(0)

    def predict_stars(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        if self.stars is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.stars_tok = AutoTokenizer.from_pretrained(self._stars_name)
            self.stars = AutoModelForSequenceClassification.from_pretrained(self._stars_name).to(self.device).eval()
        # labels look like "1 star", "2 stars", ... -> star value per class index
        star_vals = np.array([int(re.search(r"\d", str(self.stars.config.id2label[i])).group())
                              for i in range(self.stars.config.num_labels)], dtype=float)
        preds = []
        texts = list(texts)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                enc = self.stars_tok(texts[start:start + self.batch_size], padding=True,
                                     truncation=True, max_length=512, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.stars(**enc).logits, dim=-1).cpu().numpy()
                preds.append(star_vals[probs.argmax(1)])
        return np.concatenate(preds) if preds else np.zeros(0)


def get_engine(preferred: str = "auto"):
    """'hf', 'lite' or 'auto' (hf if transformers + model download work, else lite)."""
    if preferred in ("hf", "auto"):
        try:
            return HFEngine()
        except Exception as e:  # no internet, no torch, etc.
            if preferred == "hf":
                raise
            print(f"[ReviewRadar] Hugging Face engine unavailable ({type(e).__name__}: {e}). Using lite engine.")
    return LiteEngine()


# --------------------------------------------------------------------------------------
# 3. Lite star model (supervised learning on the Kaggle dataset)
# --------------------------------------------------------------------------------------
def train_star_model(data_path: str = DATA_PATH, save_path: str = STAR_MODEL_PATH, verbose=True):
    """TF-IDF (1-2 grams) + multinomial logistic regression, 80/20 split, seed 42."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error
    import joblib

    df = pd.read_csv(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(df["Review"], df["Rating"], test_size=0.2,
                                              random_state=SEED, stratify=df["Rating"])
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "test_size": int(len(y_te)),
        "accuracy_exact": round(float(accuracy_score(y_te, pred)), 3),
        "accuracy_within_1_star": round(float(np.mean(np.abs(pred - y_te) <= 1)), 3),
        "mae_stars": round(float(mean_absolute_error(y_te, pred)), 3),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    with open(save_path.replace(".joblib", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print("[ReviewRadar] star model trained:", metrics)
    return model


def load_or_train_star_model():
    import joblib
    if os.path.exists(STAR_MODEL_PATH):
        try:
            return joblib.load(STAR_MODEL_PATH)
        except Exception:
            pass  # e.g. scikit-learn version mismatch: retrain
    return train_star_model()


# --------------------------------------------------------------------------------------
# 4. Analytics
# --------------------------------------------------------------------------------------
STOPWORDS = set("""a an the and or of to in on at for with is was were be been are it its this that
very so too just not no n't t i we my our you they he she them us me had has have did do does got
get would could will there here from as by about after before up out all one only also really
stay stayed hotel hotels night nights day days time place good great nice bad didn don like went even
booked back said told people little bit lot way thing things think going make made want wanted""".split())


@dataclass
class Results:
    engine_label: str
    reviews: pd.DataFrame
    mentions: pd.DataFrame
    summary: pd.DataFrame
    kpis: dict
    evidence: dict = field(default_factory=dict)
    complaint_terms: dict = field(default_factory=dict)


_ALL_ASPECT_WORDS = set(w for spec in ASPECTS.values() for kw in spec["keywords"] for w in kw.split())
_ALL_ASPECT_WORDS |= {w + "s" for w in _ALL_ASPECT_WORDS}
_PLACE_WORDS = set("""paris london amsterdam barcelona new york berlin rome seattle boston san francisco
punta cana hawaii waikiki europe france french euro euros""".split())


def _complaint_terms(neg_clauses: pd.Series, aspect: str, k: int = 6) -> list[str]:
    """Most frequent descriptive words in complaints (what exactly is wrong)."""
    cnt = Counter()
    for c in neg_clauses:
        toks = set(w for w in re.findall(r"[a-z']+", c.lower())
                   if len(w) > 2 and w not in STOPWORDS and w not in _ALL_ASPECT_WORDS and w not in _PLACE_WORDS)
        cnt.update(toks)  # count once per snippet
    return [w for w, n in cnt.most_common(k) if n >= 2]


def analyze(df: pd.DataFrame, engine, text_col: str = "Review", rating_col: str | None = "Rating",
            max_reviews: int | None = None, progress=None) -> Results:
    """Full ReviewRadar pipeline on a dataframe of reviews."""
    data = df.copy()
    data = data[data[text_col].notna() & (data[text_col].astype(str).str.strip() != "")]
    if max_reviews and len(data) > max_reviews:
        data = data.sample(max_reviews, random_state=SEED)
    data = data.reset_index(drop=True)
    texts = data[text_col].astype(str)

    # Ratings: use real ones when provided, otherwise predict them
    if rating_col and rating_col in data.columns and pd.to_numeric(data[rating_col], errors="coerce").notna().mean() > 0.8:
        rating = pd.to_numeric(data[rating_col], errors="coerce")
        rating = rating.fillna(rating.median())
        rating_source = "guest rating"
    else:
        if progress:
            progress(0.05, desc="Predicting star ratings")
        rating = pd.Series(engine.predict_stars(texts.tolist()), index=data.index)
        rating_source = "predicted by AI"

    # Aspect mentions + sentiment
    mentions = extract_mentions(texts)
    if len(mentions):
        mentions["polarity"] = engine.score_pairs(mentions["clause"].tolist(), mentions["term"].tolist(),
                                                  progress=progress)
    else:
        mentions["polarity"] = []
    mentions["label"] = np.where(mentions["polarity"] > 0.05, "positive",
                                 np.where(mentions["polarity"] < -0.05, "negative", "neutral"))

    # Review x aspect matrix (mean polarity, 0 = not mentioned)
    mat = (mentions.groupby(["review_id", "aspect"])["polarity"].mean().unstack()
           .reindex(index=data.index, columns=ASPECT_NAMES))
    mentioned = mat.notna()
    X = mat.fillna(0.0)

    # Review-level flags: did the guest complain about / praise each aspect?
    neg_flag = (mentions[mentions["label"] == "negative"].groupby(["review_id", "aspect"]).size()
                .unstack().reindex(index=data.index, columns=ASPECT_NAMES).notna().astype(float))
    pos_flag = (mentions[mentions["label"] == "positive"].groupby(["review_id", "aspect"]).size()
                .unstack().reindex(index=data.index, columns=ASPECT_NAMES).notna().astype(float))

    # Driver analysis (supervised regression, controls for co-mentions):
    # rating ~ b0 + sum_a( penalty_a * complained_a + bonus_a * praised_a )
    penalty = pd.Series(np.nan, index=ASPECT_NAMES)
    bonus = pd.Series(np.nan, index=ASPECT_NAMES)
    r2 = None
    if len(data) >= 40 and rating.std() > 0:
        from sklearn.linear_model import Ridge
        feats = np.hstack([neg_flag.values, pos_flag.values])
        reg = Ridge(alpha=1.0).fit(feats, rating.values)
        k = len(ASPECT_NAMES)
        penalty = pd.Series(-reg.coef_[:k], index=ASPECT_NAMES).clip(lower=0)  # stars lost per complaint
        bonus = pd.Series(reg.coef_[k:], index=ASPECT_NAMES).clip(lower=0)     # stars gained per praise
        r2 = float(reg.score(feats, rating.values))

    rows = []
    for a in ASPECT_NAMES:
        m = mentions[mentions["aspect"] == a]
        n_rev = int(mentioned[a].sum())
        neg_rev = neg_flag[a] > 0
        pos_rev = (pos_flag[a] > 0) & ~neg_rev
        pos, neg = int((m["label"] == "positive").sum()), int((m["label"] == "negative").sum())
        net = 100.0 * (pos - neg) / len(m) if len(m) else np.nan
        gap = (rating[pos_rev].mean() - rating[neg_rev].mean()) if pos_rev.sum() >= 3 and neg_rev.sum() >= 3 else np.nan
        pen, bon = penalty[a], bonus[a]
        if np.isnan(pen):  # small samples: fall back on the raw rating gap
            pen = max(0.0, gap) if not np.isnan(gap) else 0.0
            bon = 0.0
        imp = max(0.0, float(pen + bon))  # swing in stars between a complaint and a praise
        # expected average star gain if every complaint about this aspect disappeared
        gain = max(0.0, float(pen * neg_rev.mean()))
        rows.append({
            "aspect": a,
            "mentions": int(len(m)),
            "review_share_%": round(100.0 * n_rev / max(1, len(data)), 1),
            "positive": pos, "negative": neg, "neutral": int(len(m) - pos - neg),
            "net_sentiment": round(net, 1) if not np.isnan(net) else np.nan,
            "negative_review_share_%": round(100.0 * neg_rev.sum() / max(1, len(data)), 1),
            "rating_gap_stars": round(gap, 2) if not np.isnan(gap) else np.nan,
            "stars_lost_per_complaint": round(float(pen), 3),
            "impact_stars": round(float(imp), 3),
            "star_gain_if_fixed": round(gain, 3),
        })
    summary = pd.DataFrame(rows).sort_values("star_gain_if_fixed", ascending=False).reset_index(drop=True)
    summary.insert(0, "priority", range(1, len(summary) + 1))

    # Evidence: most negative / most positive clauses per aspect
    evidence, complaint_terms = {}, {}
    for a in ASPECT_NAMES:
        m = mentions[mentions["aspect"] == a].drop_duplicates("clause")
        evidence[a] = {
            "negative": m.nsmallest(3, "polarity").query("polarity < -0.05")["clause"].tolist(),
            "positive": m.nlargest(3, "polarity").query("polarity > 0.05")["clause"].tolist(),
        }
        complaint_terms[a] = _complaint_terms(m[m["label"] == "negative"]["clause"], a)

    reviews_out = data[[text_col]].rename(columns={text_col: "review"}).copy()
    reviews_out["rating"] = rating.round(2)
    for a in ASPECT_NAMES:
        reviews_out[a] = mat[a].round(2)

    top = summary.iloc[0]
    kpis = {
        "engine": engine.label,
        "reviews_analyzed": int(len(data)),
        "aspect_mentions": int(len(mentions)),
        "average_rating": round(float(rating.mean()), 2),
        "rating_source": rating_source,
        "top_issue": top["aspect"],
        "top_issue_star_gain": float(top["star_gain_if_fixed"]),
        "total_star_gain_top3": round(float(summary["star_gain_if_fixed"].head(3).sum()), 3),
        "driver_model_r2": round(r2, 3) if r2 is not None else None,
    }
    if progress:
        progress(1.0, desc="Done")
    return Results(engine.label, reviews_out, mentions, summary, kpis, evidence, complaint_terms)


# --------------------------------------------------------------------------------------
# 5. Business value: illustrative revenue upside
# --------------------------------------------------------------------------------------
PRICE_PREMIUM_PER_STAR = 11.2  # Anderson (2012), Cornell CHR: +1 point on a 5-point review scale -> +11.2% price, same occupancy
REVPAR_ELASTICITY = 1.42       # Anderson (2012): +1% reputation index -> up to +1.42% RevPAR (shown as a sensitivity only)


def revenue_upside(current_rating: float, star_gain: float, rooms: int, adr: float, occupancy_pct: float,
                   complaint_reduction_pct: float = 50.0,
                   premium_per_star: float = PRICE_PREMIUM_PER_STAR) -> dict:
    """Illustrative annual room-revenue upside of a rating improvement.

    star_gain is the gain if ALL complaints on the chosen aspects disappeared; a realistic plan removes only part
    of them (complaint_reduction_pct). The extra rating is converted into pricing power at constant occupancy
    with the Travelocity estimate of Anderson (2012): +11.2% room rate per review point.
    """
    current_rating = max(1.0, float(current_rating))
    realised_gain = star_gain * complaint_reduction_pct / 100.0
    new_rating = min(5.0, current_rating + realised_gain)
    realised_gain = new_rating - current_rating
    revpar = adr * occupancy_pct / 100.0
    revpar_uplift_pct = premium_per_star * realised_gain
    annual_room_revenue = rooms * 365 * revpar
    reputation_change_pct = 100.0 * realised_gain / current_rating
    return {
        "complaint_reduction_pct": complaint_reduction_pct,
        "star_gain": round(realised_gain, 2),
        "new_rating": round(new_rating, 2),
        "reputation_change_pct": round(reputation_change_pct, 2),
        "revpar": round(revpar, 2),
        "revpar_uplift_pct": round(revpar_uplift_pct, 2),
        "annual_room_revenue": round(annual_room_revenue),
        "annual_upside": round(annual_room_revenue * revpar_uplift_pct / 100.0),
        # sensitivity with the reputation-index elasticity (upper bound)
        "revpar_uplift_pct_upper": round(REVPAR_ELASTICITY * reputation_change_pct, 2),
    }


# --------------------------------------------------------------------------------------
# 6. Action plan (template) and optional LLM memo
# --------------------------------------------------------------------------------------
def action_plan_markdown(res: Results, hotel_name: str = "your hotel", top_n: int = 3,
                         upside: dict | None = None) -> str:
    k = res.kpis
    lines = [f"## ReviewRadar action plan for {hotel_name}",
             f"*{k['reviews_analyzed']} reviews analysed, average rating {k['average_rating']} / 5 "
             f"({k['rating_source']}), engine: {k['engine']}.*", ""]
    for _, r in res.summary.head(top_n).iterrows():
        a = r["aspect"]
        lines.append(f"### Priority {int(r['priority'])}: {a}")
        lines.append(f"- {r['negative_review_share_%']}% of reviews complain about it; net sentiment {r['net_sentiment']}.")
        if not pd.isna(r["rating_gap_stars"]):
            lines.append(f"- Guests who complain about {a.lower()} rate the stay {r['rating_gap_stars']} stars lower than guests who praise it.")
        lines.append(f"- Each complaint about it costs about {r['stars_lost_per_complaint']:.2f} stars; "
                     f"removing these complaints could add +{r['star_gain_if_fixed']:.2f} stars to the average rating.")
        terms = res.complaint_terms.get(a) or []
        if terms:
            lines.append(f"- Recurring words in complaints: {', '.join(terms)}.")
        ev = res.evidence[a]["negative"][:2]
        if ev:
            lines.append("- What guests say: " + " / ".join(f"\"{e}\"" for e in ev))
        lines.append("- Recommended actions:")
        for act in ASPECTS[a]["actions"]:
            lines.append(f"  - {act}")
        lines.append("")
    strengths = res.summary.dropna(subset=["net_sentiment"]).query("mentions >= 5").nlargest(2, "net_sentiment")
    if len(strengths):
        lines.append("### Strengths to promote in marketing")
        for _, r in strengths.iterrows():
            ev = res.evidence[r["aspect"]]["positive"][:1]
            quote = f" (\"{ev[0]}\")" if ev else ""
            lines.append(f"- {r['aspect']}: net sentiment +{r['net_sentiment']}{quote}")
        lines.append("")
    if upside:
        lines.append("### Business case (illustrative)")
        lines.append(f"- If the plan removes {upside['complaint_reduction_pct']:.0f}% of the complaints on the top {top_n} aspects, "
                     f"the average rating moves from {k['average_rating']} to {upside['new_rating']} (+{upside['star_gain']} stars).")
        lines.append(f"- At +11.2% room rate per review point (Anderson, 2012), that is +{upside['revpar_uplift_pct']}% RevPAR, "
                     f"about EUR {upside['annual_upside']:,.0f} per year on EUR {upside['annual_room_revenue']:,.0f} of room revenue.")
        lines.append(f"- Sensitivity: with the reputation-index elasticity of the same study, the upside could reach "
                     f"+{upside['revpar_uplift_pct_upper']}% RevPAR (upper bound).")
    return "\n".join(lines)


def llm_memo(res: Results, hotel_name: str = "the hotel", model_id: str = HF_LLM_MODEL,
             max_new_tokens: int = 350) -> str:
    """Optional: a small open LLM writes a 150-word memo to the General Manager from the structured results."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.half().to("cuda")
    model.eval()
    facts = {
        "hotel": hotel_name,
        "average_rating": res.kpis["average_rating"],
        "top_issues": [
            {"aspect": r["aspect"], "negative_review_share_%": r["negative_review_share_%"],
             "star_gain_if_fixed": r["star_gain_if_fixed"],
             "guest_quotes": res.evidence[r["aspect"]]["negative"][:2]}
            for _, r in res.summary.head(3).iterrows()
        ],
    }
    messages = [
        {"role": "system", "content": "You are a hotel operations consultant. Be concrete, concise and factual. Use only the data given."},
        {"role": "user", "content": "Write a memo of maximum 150 words to the General Manager with the 3 priorities, "
                                    "one concrete action each and the expected rating gain. Data:\n" + json.dumps(facts)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------
# 7. Market benchmark (investor / asset-manager view)
# --------------------------------------------------------------------------------------
DEFAULT_MARKETS = ["Paris", "London", "Amsterdam", "Barcelona", "New York", "Berlin"]


def market_segment(df: pd.DataFrame, city: str, text_col: str = "Review") -> pd.DataFrame:
    return df[df[text_col].str.contains(r"\b" + re.escape(city) + r"\b", case=False, na=False)]


def benchmark(df: pd.DataFrame, engine, cities=DEFAULT_MARKETS, per_city: int = 250, progress=None):
    """Net sentiment per aspect for reviews mentioning each city. Returns (matrix, sizes)."""
    cols, sizes = {}, {}
    for i, city in enumerate(cities):
        seg = market_segment(df, city)
        if len(seg) < 20:
            continue
        if progress:
            progress(i / len(cities), desc=f"Benchmarking {city}")
        res = analyze(seg, engine, max_reviews=per_city)
        cols[city] = res.summary.set_index("aspect")["net_sentiment"]
        sizes[city] = {"reviews": res.kpis["reviews_analyzed"], "average_rating": res.kpis["average_rating"]}
    mat = pd.DataFrame(cols).reindex(ASPECT_NAMES)
    return mat, sizes


# --------------------------------------------------------------------------------------
# 8. Charts (Plotly)
# --------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13)


def _layout(fig, title, height=420):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), font=FONT, height=height,
                      plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                      margin=dict(l=10, r=20, t=60, b=40), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", color=INK2)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", color=INK2)
    return fig


def fig_aspect_sentiment(res: Results):
    """Diverging bars: share of negative mentions (left, red) vs positive mentions (right, blue)."""
    import plotly.graph_objects as go
    s = res.summary[res.summary["mentions"] > 0].copy()
    s["pos_%"] = 100.0 * s["positive"] / s["mentions"]
    s["neg_%"] = 100.0 * s["negative"] / s["mentions"]
    s = s.sort_values("neg_%")  # most complained-about aspect on top
    hover = "<b>%{y}</b><br>%{customdata[0]:.0f}% positive, %{customdata[1]:.0f}% negative<br>%{customdata[2]} mentions<extra></extra>"
    cd = np.stack([s["pos_%"], s["neg_%"], s["mentions"]], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=-s["neg_%"], y=s["aspect"], orientation="h", name="Negative",
                         marker=dict(color=NEG, cornerradius=4), customdata=cd, hovertemplate=hover,
                         text=[f"{v:.0f}%" for v in s["neg_%"]], textposition="outside", textfont=dict(color=INK2)))
    fig.add_trace(go.Bar(x=s["pos_%"], y=s["aspect"], orientation="h", name="Positive",
                         marker=dict(color=POS, cornerradius=4), customdata=cd, hovertemplate=hover,
                         text=[f"{v:.0f}%" for v in s["pos_%"]], textposition="outside", textfont=dict(color=INK2)))
    lim = float(max(s["pos_%"].max(), s["neg_%"].max())) + 18
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_xaxes(range=[-lim, lim], tickvals=[-75, -50, -25, 0, 25, 50, 75],
                     ticktext=["75%", "50%", "25%", "0", "25%", "50%", "75%"],
                     title="Share of mentions: negative (left) vs positive (right)")
    _layout(fig, "What guests love and hate, by aspect")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.08,
                                                   traceorder="normal", font=dict(color=INK2)))
    return fig


def _place_labels(xs, ys, sizes, texts, xr, yr, W=620.0, H=320.0, pad=6.0):
    """Greedy label placement in pixel space: pick the side (top/bottom/right/left) that overlaps least."""
    def to_px(x, y):
        return (x - xr[0]) / (xr[1] - xr[0]) * W, (y - yr[0]) / (yr[1] - yr[0]) * H

    def overlap(a, b):
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        return max(0, dx) * max(0, dy)

    pts = [to_px(x, y) for x, y in zip(xs, ys)]
    obstacles = [(px - r, py - r, px + r, py + r) for (px, py), r in zip(pts, np.asarray(sizes) / 2)]
    placed, shifts = [], []
    order = np.argsort(-np.asarray(ys))
    result = [None] * len(xs)
    for i in order:
        (px, py), r, t = pts[i], sizes[i] / 2, texts[i]
        w, h = 7.2 * len(t), 16
        cands = {
            "top": (0, r + 4 + h / 2), "bottom": (0, -(r + 4 + h / 2)),
            "right": (r + 6 + w / 2, 0), "left": (-(r + 6 + w / 2), 0),
        }
        best, best_cost = None, None
        for k, (dx, dy) in cands.items():
            cx, cy = px + dx, py + dy
            box = (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)
            cost = sum(overlap(box, b) for b in placed) + sum(overlap(box, o) for o in obstacles)
            cost += 1e4 * (box[0] < 0 or box[2] > W or box[1] < 0 or box[3] > H)
            cost += {"top": 0, "bottom": 1, "right": 2, "left": 3}[k]  # preference order
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy, box), cost
        placed.append(best[2])
        result[i] = (best[0], best[1])
    return result


def fig_ipa(res: Results):
    """Importance-Performance Analysis (Martilla & James, 1977)."""
    import plotly.graph_objects as go
    s = res.summary.dropna(subset=["net_sentiment"]).copy()
    x_mid, y_mid = float(s["net_sentiment"].median()), float(s["impact_stars"].median())
    fix = (s["net_sentiment"] < x_mid) & (s["impact_stars"] >= y_mid)
    s, fix = s.assign(_f=fix).sort_values("_f"), fix.loc[s.assign(_f=fix).sort_values("_f").index]  # red on top
    sizes = np.clip(np.sqrt(s["mentions"].values) * 1.6, 12, 40)
    xr = [min(-10, s["net_sentiment"].min() - 15), min(105, s["net_sentiment"].max() + 15)]
    yr = [0, max(0.1, s["impact_stars"].max() * 1.3)]
    fig = go.Figure(go.Scatter(
        x=s["net_sentiment"], y=s["impact_stars"], mode="markers", text=s["aspect"],
        marker=dict(size=sizes, color=[NEG if f else POS for f in fix], opacity=0.88,
                    line=dict(color="#fcfcfb", width=2)),
        customdata=np.stack([s["mentions"], s["star_gain_if_fixed"]], axis=-1),
        hovertemplate="<b>%{text}</b><br>Performance (net sentiment) %{x:+.1f}<br>"
                      "Impact on rating %{y:.2f} stars<br>%{customdata[0]} mentions<br>"
                      "Star gain if fixed +%{customdata[1]:.2f}<extra></extra>",
    ))
    shifts = _place_labels(s["net_sentiment"].values, s["impact_stars"].values, sizes, s["aspect"].tolist(), xr, yr)
    for (x, y, t, f), (dx, dy) in zip(zip(s["net_sentiment"], s["impact_stars"], s["aspect"], fix), shifts):
        fig.add_annotation(x=x, y=y, text=f"<b>{t}</b>" if f else t, showarrow=False, xshift=dx, yshift=dy,
                           font=dict(color=INK if f else INK2, size=12))
    fig.add_vline(x=x_mid, line=dict(color="#c3c2b7", dash="dot"))
    fig.add_hline(y=y_mid, line=dict(color="#c3c2b7", dash="dot"))
    for txt, x, y, anchor in [("FIX FIRST", xr[0], yr[1], "left"), ("KEEP UP THE GOOD WORK", xr[1], yr[1], "right"),
                              ("LOW PRIORITY", xr[0], 0, "left"), ("POSSIBLE OVERKILL", xr[1], 0, "right")]:
        fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False, xanchor=anchor,
                           yanchor="top" if y else "bottom", font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=xr, zeroline=False, title="Performance: net sentiment (higher is better)")
    fig.update_yaxes(range=yr, title="Importance: impact on rating (stars)")
    return _layout(fig, "Importance-Performance map: where to invest first", height=460)


def fig_benchmark(mat: pd.DataFrame):
    """Colour = net sentiment relative to the average of the selected markets; text = absolute net sentiment."""
    import plotly.graph_objects as go
    delta = mat.sub(mat.mean(axis=1), axis=0)
    lim = float(np.nanmax(np.abs(delta.values))) if delta.size else 10.0
    lim = max(5.0, math.ceil(lim / 5) * 5)
    fig = go.Figure(go.Heatmap(
        z=delta.values, x=mat.columns, y=mat.index, zmin=-lim, zmax=lim, zmid=0,
        colorscale=[[0, NEG], [0.5, MID], [1, POS]],
        text=[[("" if pd.isna(v) else f"{v:+.0f}") for v in row] for row in mat.values], texttemplate="%{text}",
        textfont=dict(color=INK), xgap=2, ygap=2, customdata=mat.values,
        hovertemplate="<b>%{x}</b> / %{y}<br>Net sentiment %{customdata:+.1f}<br>"
                      "%{z:+.1f} pts vs average of these markets<extra></extra>",
        colorbar=dict(title=dict(text="vs market<br>average (pts)"), outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    _layout(fig, "Market benchmark: net guest sentiment by city (colour = better or worse than average)", height=470)
    fig.update_layout(margin=dict(l=10, r=20, t=90, b=20))
    return fig


# --------------------------------------------------------------------------------------
# 9. Evaluation (how much can we trust the engine?)
# --------------------------------------------------------------------------------------
def held_out_test_set(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Same 80/20 split as the star model, so the test reviews were never seen in training."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(data_path)
    _, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["Rating"])
    return test


def evaluate_engine(engine, n: int = 500, data_path: str = DATA_PATH) -> dict:
    """Checks the engine against the real guest ratings on n held-out reviews."""
    from scipy.stats import spearmanr
    test = held_out_test_set(data_path).sample(n, random_state=SEED)
    res = analyze(test, engine, max_reviews=None)
    r = res.reviews
    mean_pol = r[ASPECT_NAMES].mean(axis=1, skipna=True)
    ok = mean_pol.notna()
    rho = spearmanr(mean_pol[ok], r.loc[ok, "rating"]).correlation
    low = r["rating"] <= 2
    neg_any = (r[ASPECT_NAMES] < -0.05).any(axis=1)
    stars = engine.predict_stars(test["Review"].tolist())
    y = test["Rating"].values
    return {
        "engine": engine.label,
        "n_reviews": int(n),
        "aspect_sentiment_vs_rating_spearman": round(float(rho), 3),
        "share_of_1_2_star_reviews_with_a_detected_complaint_%": round(100.0 * neg_any[low].mean(), 1),
        "star_prediction_accuracy_exact": round(float(np.mean(stars == y)), 3),
        "star_prediction_accuracy_within_1": round(float(np.mean(np.abs(stars - y) <= 1)), 3),
        "star_prediction_mae": round(float(np.mean(np.abs(stars - y))), 3),
    }
