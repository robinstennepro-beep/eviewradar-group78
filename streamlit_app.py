"""
ReviewRadar: Streamlit web app (permanent hosted version)
Group 78 · Advanced AI Algorithms · ESSEC MSc IMHI

Run locally:   streamlit run streamlit_app.py
Hosted:        Streamlit Community Cloud, deployed from the GitHub repository (main file: streamlit_app.py)

Same engine as the Gradio app (reviewradar_core.py). The hosted version uses the lite engine
(VADER + TF-IDF model) because the free cloud server has no GPU. Set REVIEWRADAR_ENGINE=auto
on a machine with a GPU to use the Hugging Face transformer engine.
"""

import io
import os
import zipfile

import pandas as pd
import streamlit as st

import reviewradar_core as rr

st.set_page_config(page_title="ReviewRadar · Group 78", page_icon="📡", layout="wide")

ENGINE_MODE = os.environ.get("REVIEWRADAR_ENGINE", "lite")  # lite | auto | hf
VIDEO_PATH = os.path.join(rr.HERE, "assets", "ReviewRadar_demo_Group78.mp4")

DEMO_SETS = {
    "Demo 1: Punta Cana resort reviews (1,142 reviews)": ("city", "Punta Cana"),
    "Demo 2: Paris hotel reviews (744 reviews)": ("city", "Paris"),
    "Demo 3: Random sample of the full TripAdvisor dataset": ("random", None),
}

# --------------------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------------------
LOGO_SVG = """<svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="22" cy="22" r="20" stroke="#8fc0ff" stroke-width="2.5" opacity=".5"/>
<circle cx="22" cy="22" r="13" stroke="#8fc0ff" stroke-width="2.5" opacity=".8"/>
<circle cx="22" cy="22" r="6" fill="#8fc0ff"/>
<path d="M22 22 L38 9" stroke="#ff7b7a" stroke-width="3" stroke-linecap="round"/>
<circle cx="33" cy="13" r="3" fill="#ff7b7a"/></svg>"""

st.markdown(
    """
<style>
.block-container {max-width: 1480px; padding-top: 3rem;}
.rr-header {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:18px 22px; border-radius:14px; background:linear-gradient(120deg,#0d366b 0%,#1c5cab 100%);
  color:#fff; margin-bottom:8px;}
.rr-header * {color:#fff !important;}
.rr-brand {display:flex; align-items:center; gap:14px;}
.rr-name {font-size:28px; font-weight:700; letter-spacing:-0.02em; line-height:1.1;}
.rr-tag {font-size:14px; opacity:.85; margin-top:3px;}
.rr-badges {display:flex; gap:8px; flex-wrap:wrap;}
.rr-badges span {font-size:12px; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.25);}
.rr-kpis {display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px;}
.rr-card {flex:1; min-width:160px; padding:14px 16px; border:1px solid rgba(11,11,11,.1); border-radius:10px;
  background:#fcfcfb;}
.rr-card .l {font-size:12px; color:#52514e;}
.rr-card .v {font-size:26px; font-weight:650; color:#0b0b0b; margin-top:2px;}
.rr-card .s {font-size:12px; color:#898781;}
.rr-empty {padding:48px 24px; border:1.5px dashed #c3c2b7; border-radius:12px; text-align:center; color:#52514e;}
.rr-empty b.t {display:block; font-size:18px; color:#0b0b0b; margin-bottom:6px;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:#1c5cab; border-color:#1c5cab; color:#fff;}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  background:#184f95; border-color:#184f95;}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="rr-header">
  <div class="rr-brand">{LOGO_SVG}
    <div><div class="rr-name">ReviewRadar</div>
    <div class="rr-tag">Guest reviews in, prioritised action plan and revenue business case out.</div></div>
  </div>
  <div class="rr-badges"><span>Aspect-based sentiment</span><span>Driver analysis</span><span>Revenue business case</span></div>
</div>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the AI engine (first visit only, about 30 seconds)")
def get_engine(mode: str):
    eng = rr.get_engine(mode)
    if eng.name == "lite":
        eng.predict_stars(["warm-up"])  # trains or loads the star model once per server
    return eng


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return pd.read_csv(rr.DATA_PATH)


class StreamlitProgress:
    """Adapter so reviewradar_core can report progress (it calls progress(fraction, desc=...))."""

    def __init__(self):
        self.bar = st.progress(0.0, text="Starting")

    def __call__(self, frac, desc=""):
        self.bar.progress(float(min(1.0, max(0.0, frac))), text=desc or "Working")

    def done(self):
        self.bar.empty()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def guess_columns(df: pd.DataFrame):
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in cols if any(k in c for k in ("review", "text", "comment", "content", "avis"))), None)
    if text_col is None:
        obj = [c for c in df.columns if df[c].dtype == object]
        if not obj:
            return None, None
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    rating_col = next((cols[c] for c in cols if any(k in c for k in ("rating", "score", "star", "note"))), None)
    return text_col, rating_col


def load_input(demo_choice, upload, pasted):
    if upload is not None:
        df = pd.read_csv(upload)
        text_col, rating_col = guess_columns(df)
        if text_col is None:
            raise ValueError("No text column found in the CSV.")
        return df, text_col, rating_col, upload.name
    if pasted and pasted.strip():
        lines = [l.strip() for l in pasted.splitlines() if len(l.strip().split()) >= 3]
        return pd.DataFrame({"Review": lines}), "Review", None, "pasted reviews"
    kind, value = DEMO_SETS[demo_choice]
    df = get_dataset()
    if kind == "city":
        return rr.market_segment(df, value), "Review", "Rating", demo_choice
    return df.sample(800, random_state=rr.SEED), "Review", "Rating", demo_choice


def kpi_html(k, up):
    def card(label, value, sub=""):
        return f"<div class='rr-card'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>"

    return (
        "<div class='rr-kpis'>"
        + card("Reviews analysed", f"{k['reviews_analyzed']:,}", f"{k['aspect_mentions']:,} aspect mentions")
        + card("Average rating", f"{k['average_rating']} / 5", k["rating_source"])
        + card("Priority #1", k["top_issue"], f"+{k['top_issue_star_gain']:.2f} stars if fixed")
        + card("Revenue upside", f"+{up['revpar_uplift_pct']}% RevPAR",
               f"about EUR {up['annual_upside']:,.0f} / year (illustrative)")
        + "</div>"
    )


def results_zip(res, plan: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("aspect_summary.csv", res.summary.to_csv(index=False))
        z.writestr("aspect_mentions.csv", res.mentions.to_csv(index=False))
        z.writestr("review_scores.csv", res.reviews.to_csv(index=False))
        z.writestr("action_plan.md", plan)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1. Analyse a hotel", "2. Market benchmark", "3. Demo video", "4. Method & sources"])

with tab1:
    left, right = st.columns([1, 2], gap="large")
    with left:
        with st.form("analyse"):
            demo_choice = st.selectbox("Demo data", list(DEMO_SETS))
            upload = st.file_uploader("...or upload your own CSV (one review per row, optional rating column)",
                                      type=["csv"])
            pasted = st.text_area("...or paste reviews (one per line)", height=110)
            hotel_name = st.text_input("Hotel name", placeholder="e.g. Hotel Demo Punta Cana")
            max_reviews = st.slider("Max reviews to analyse", 50, 3000, 600, step=50)
            with st.expander("Revenue assumptions"):
                rooms = st.number_input("Number of rooms", min_value=1, value=120, step=1)
                adr = st.number_input("ADR (EUR)", min_value=1.0, value=180.0, step=5.0)
                occupancy = st.number_input("Occupancy (%)", min_value=1.0, max_value=100.0, value=75.0, step=1.0)
                reduction = st.slider("Share of complaints the plan removes on the top 3 aspects (%)", 10, 100, 50, step=10)
            run = st.form_submit_button("Run ReviewRadar", type="primary", width="stretch")
        st.caption("Tip: try data/upload_example_40_reviews.csv from the project package to test the upload.")

    with right:
        if run:
            try:
                df, text_col, rating_col, source = load_input(demo_choice, upload, pasted)
            except Exception as e:
                st.error(f"Could not read the reviews: {e}")
                df = None
            if df is not None and len(df) == 0:
                st.error("No reviews found.")
            elif df is not None:
                eng = get_engine(ENGINE_MODE)
                prog = StreamlitProgress()
                res = rr.analyze(df, eng, text_col=text_col, rating_col=rating_col,
                                 max_reviews=int(max_reviews), progress=prog)
                prog.done()
                up = rr.revenue_upside(res.kpis["average_rating"], res.kpis["total_star_gain_top3"],
                                       int(rooms), float(adr), float(occupancy), float(reduction))
                name = hotel_name.strip() or source
                st.session_state["analysis"] = {
                    "res": res, "up": up, "plan": rr.action_plan_markdown(res, name, upside=up),
                    "engine": eng.label,
                }

        a = st.session_state.get("analysis")
        if a is None:
            st.markdown("<div class='rr-empty'><b class='t'>No analysis yet</b>Pick a demo data set or upload "
                        "your reviews, then click <b>Run ReviewRadar</b>.</div>", unsafe_allow_html=True)
        else:
            res, up, plan = a["res"], a["up"], a["plan"]
            if res.kpis["reviews_analyzed"] < 50:
                st.warning("Fewer than 50 reviews: the sentiment bars are indicative, but the star impact, the "
                           "priorities and the business case need at least 50 to 100 reviews to be meaningful.")
            st.markdown(kpi_html(res.kpis, up), unsafe_allow_html=True)
            st.plotly_chart(rr.fig_aspect_sentiment(res), width="stretch")
            st.plotly_chart(rr.fig_ipa(res), width="stretch")
            with st.container(border=True):
                st.markdown(plan)
            st.subheader("Aspect scorecard")
            table = res.summary[["priority", "aspect", "mentions", "net_sentiment", "negative_review_share_%",
                                 "stars_lost_per_complaint", "star_gain_if_fixed"]]
            st.dataframe(table, hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            c1.download_button("Download all results (zip)", results_zip(res, plan),
                               file_name="reviewradar_results.zip", mime="application/zip",
                               width="stretch")
            c2.download_button("Download the action plan (Markdown)", plan, file_name="action_plan.md",
                               mime="text/markdown", width="stretch")
            st.caption(f"Engine: {a['engine']}")

with tab2:
    st.markdown("Compare guest sentiment across destinations: useful for investors, asset managers and brands "
                "entering a market.")
    cities = st.multiselect("Markets", rr.DEFAULT_MARKETS, default=rr.DEFAULT_MARKETS)
    per_city = st.slider("Reviews per market", 50, 800, 250, step=50)
    if st.button("Run benchmark", type="primary"):
        if not cities:
            st.error("Pick at least one city.")
        else:
            key = (tuple(cities), int(per_city), ENGINE_MODE)
            cache = st.session_state.setdefault("bench_cache", {})
            if key not in cache:
                eng = get_engine(ENGINE_MODE)
                prog = StreamlitProgress()
                cache[key] = rr.benchmark(get_dataset(), eng, cities, int(per_city), progress=prog)
                prog.done()
            st.session_state["bench_key"] = key
    bkey = st.session_state.get("bench_key")
    if bkey:
        mat, sizes = st.session_state["bench_cache"][bkey]
        st.plotly_chart(rr.fig_benchmark(mat), width="stretch")
        info = pd.DataFrame(sizes).T.reset_index().rename(columns={"index": "market"})
        info["reviews"] = info["reviews"].astype(int)
        st.dataframe(info, hide_index=True, width="stretch")

with tab3:
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.info("The demo video is part of the project package (ReviewRadar_demo_Group78.mp4).")

with tab4:
    st.markdown("""
### How ReviewRadar works
1. **Opinion mining**: every review is cut into short opinion snippets; each snippet is mapped to one of 8 hospitality
   aspects (Room & Bed, Cleanliness, Staff & Service, Location & View, Food & Beverage, Noise & Sleep, Value & Price,
   Amenities & Wi-Fi).
2. **Aspect-based sentiment**: on a GPU, a DeBERTa-v3 model fine-tuned for aspect-based sentiment
   (`yangheng/deberta-v3-base-absa-v1.1`) scores each (snippet, aspect) pair. This hosted version runs on a free
   CPU server and uses the **lite engine**: the VADER sentiment lexicon for snippets and a TF-IDF + logistic
   regression star model trained on the Kaggle data.
3. **Driver analysis (supervised learning)**: a ridge regression of the star rating on "complained about X" and
   "praised X" flags estimates how many stars each complaint really costs, controlling for the other aspects.
4. **Importance-Performance Analysis** (Martilla & James, 1977) places each aspect on a 2x2 map to show where to
   invest first.
5. **Business case**: the expected rating gain (for the share of complaints the plan removes) is converted into
   pricing power with Anderson (2012, Cornell CHR): +1 point on a 5-point review scale allows +11.2% room rate at
   the same occupancy. A sensitivity uses the same study's +1.42% RevPAR per +1% reputation index. Illustrative,
   not a forecast.

The full Hugging Face engine and the optional GM memo written by an open LLM run in the Colab notebook of the
project package.

Data: TripAdvisor Hotel Reviews (20,491 reviews), Kaggle, Alam, Ryu & Lee (2016).

*Group 78 · Advanced AI Algorithms · ESSEC MSc IMHI*
""")
