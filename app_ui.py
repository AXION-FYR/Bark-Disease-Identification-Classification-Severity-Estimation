"""
app_ui.py
=========
Streamlit front end for the Cinnamon Decision Support System demo.

Pipeline shown on screen:

    [Module 1: leaf photos]  ---\\
                                  >--  disease findings  -->  [Module 3: decision engine]  -->  report
    [Module 2: bark photos]  ---/

Module 1 and Module 2 are your teammates' CV models (each running in their
own Colab notebook, exposed over ngrok, same pattern as your decision
engine). This UI calls whichever URLs you paste into the sidebar.

WHAT MODULE 1 / 2 ARE ASSUMED TO RETURN
----------------------------------------
Per your message: "they are giving disease type, severity and percentage for
each disease." This UI accepts any of the common shapes for that so small
differences in your teammates' exact JSON won't break the demo:

    [{"disease": "leaf_blight", "severity": 40}, ...]
    [{"disease": "leaf_blight", "percentage": 40}, ...]
    {"leaf_blight": 40, "sooty_mould": 12}
    {"findings": [{"disease": "leaf_blight", "severity": 40}, ...]}

If their real response looks different, only `normalize_cv_response()` below
needs to change -- nothing else in the file.

ROBUSTNESS FOR THE LIVE DEMO
------------------------------
Network calls to two teammates' Colab notebooks are exactly the kind of
thing that can go down mid-demo. So:
  - Every remote call is wrapped and shows a clear error instead of crashing.
  - Findings are shown in an EDITABLE table before submission, so if a CV
    model is unreachable or wrong, you can type severities in by hand and
    keep the demo moving.
  - The decision engine call and the CV calls are fully independent steps.

Run with:
    pip install streamlit requests pandas
    streamlit run app_ui.py

FIX APPLIED (bark image-wise analysis not showing)
-----------------------------------------------------
Tab 2's rendering code used to live physically INSIDE the `with tab4:`
block, after an `if not report: st.stop()` guard. st.stop() halts the
entire script the instant it's called -- not just the current tab -- so
until a decision report existed, execution never reached Tab 2's code at
all, even though st.session_state.bark_image_results was already
populated correctly by Tab 1. Tab 2 is now its own top-level `with tab2:`
block (same pattern as tab1/tab3/tab4), so it renders independently of
whatever state Tab 4 is in. No data, detection, or decision-engine logic
was changed -- only where this block sits in the file.

FIX APPLIED (leaf-wise analysis missing images)
--------------------------------------------------
Tab 5 previously showed only text (a table + text-only cards) -- no leaf
photo, unlike Tab 2's bark cards which show the uploaded photo next to
its findings. Tab 1's leaf-detection handler now also stores the
uploaded leaf files into st.session_state.disease_images (same dict Tab 2
already uses for bark, with a "leaf_image_N.jpg" key prefix so it can't
collide with bark's "bark_image_N.jpg" keys). Tab 5 now renders each leaf
finding with its photo on the left and details on the right, mirroring
Tab 2 exactly. Since Module 1's /predict_full response may or may not
include which photo each finding came from, Tab 5 first checks for an
"image_name" field on the finding and falls back to positional order
(finding i -> the i-th uploaded leaf photo) if that field isn't present --
same graceful "image missing" handling Tab 2 already uses either way.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from config import API_KEY, MODULE_01_LEAF_URL, MODULE_03_DECISION_URL, REQUEST_TIMEOUT_SECONDS
from tree_severity_bridge import (
    BarkSeverityPipeline,
    load_images_from_streamlit_uploads,
)

# ---------------------------------------------------------------------------
# Constants -- must match decision_support/core.py DISEASES / PRETTY exactly,
# since the decision engine raises on any unrecognised disease name.
# ---------------------------------------------------------------------------
DISEASES = ["rough_bark", "stripecanker", "leaf_blight", "sooty_mould", "yellow_leaf_spot"]
PRETTY = {
    "rough_bark": "Rough bark",
    "stripecanker": "Stripe canker",
    "leaf_blight": "Leaf blight",
    "sooty_mould": "Sooty mould",
    "yellow_leaf_spot": "Yellow leaf spot",
}
BARK_DISEASES = {"rough_bark", "stripecanker"}
LEAF_DISEASES = {"leaf_blight", "sooty_mould", "yellow_leaf_spot"}

DISEASE_MAP = {
    "Rough bark": "rough_bark",
    "stripecanker": "stripecanker"
}

st.set_page_config(page_title="Cinnamon DSS", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------------------
# UI ONLY -- cinnamon theme (bark browns + leaf greens). No logic below
# this block changes anything about how data is fetched or computed.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root{
  --bark-900:#4A2E1E; --bark-700:#6B4423; --bark-500:#8B5E34; --bark-300:#C69A6C; --bark-100:#F0E2D0;
  --leaf-900:#243B1E; --leaf-700:#3E5C33; --leaf-500:#5C7F4A; --leaf-300:#9AB98A; --leaf-100:#E4EEDD;
  --cream:#fdf5e6;
}
.stApp{ background-color:var(--cream); }
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--bark-900) 0%, var(--bark-700) 100%);
}
section[data-testid="stSidebar"] *{ color:var(--bark-100) !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{ color:var(--bark-900) !important; }
h1,h2,h3{ color:var(--bark-900) !important; }
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:2px solid var(--bark-300); }
.stTabs [data-baseweb="tab"]{
  background-color:var(--leaf-100); border-radius:8px 8px 0 0; color:var(--bark-700);
  padding:8px 16px; font-weight:600;
}
.stTabs [aria-selected="true"]{ background-color:var(--bark-600) !important; color:black !important; }
div.stButton > button{
  background:linear-gradient(135deg, var(--bark-500), var(--bark-700));
  color:black; border:none; border-radius:8px; font-weight:600;
}
div.stButton > button:hover{ background:linear-gradient(135deg, var(--leaf-500), var(--leaf-700)); color:white; }
div[data-testid="stMetric"]{
  background-color:var(--leaf-100); border:1px solid var(--leaf-300);
  border-radius:10px; padding:12px 16px;
}
div[data-testid="stMetricLabel"]{ color:var(--bark-700) !important; }
div[data-testid="stMetricValue"]{ color:var(--bark-900) !important; }
.streamlit-expanderHeader{ background-color:var(--leaf-100); border-radius:8px; }
div[data-baseweb="notification"]{ border-radius:8px; }
.axion-hero{
  background:linear-gradient(135deg, var(--leaf-700) 0%, var(--bark-700) 100%);
  padding:24px 28px; border-radius:14px; margin-bottom:20px;
}
.axion-hero h1{ color:white !important; margin:0 0 4px 0; font-size:28px; }
.axion-hero p{ color:var(--leaf-100); margin:0; font-size:15px; }
.axion-card{
  background:white; border:1px solid var(--bark-300); border-radius:10px;
  padding:14px 16px; margin-bottom:10px;
}
.axion-badge{
  display:inline-block; padding:3px 10px; border-radius:999px;
  font-size:12px; font-weight:600; color:white;
}
</style>
""", unsafe_allow_html=True)


def stage_badge_html(stage: str | None) -> str:
    """UI ONLY -- maps a treatment stage string to a cinnamon-toned colored
    badge. Does not read, write, or alter any pipeline data; it only
    decides which color to render around whatever stage text was already
    computed elsewhere."""
    colors = {
        "Preventive": "#5C7F4A",
        "Early control": "#8B7A34",
        "Active management": "#B36A2E",
        "Severe outbreak": "#8B3A1E",
    }
    color = colors.get(stage, "#8B5E34")
    label = stage or "Unknown"
    return f'<span class="axion-badge" style="background-color:{color}">{label}</span>'


@st.cache_resource
def get_bark_pipeline():
    return BarkSeverityPipeline()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "findings" not in st.session_state:
    st.session_state.findings = []  # list of {"disease", "severity", "stage", "source"}
if "report" not in st.session_state:
    st.session_state.report = None

if "bark_image_results" not in st.session_state:
    st.session_state.bark_image_results = []

if "leaf_findings" not in st.session_state:
    st.session_state.leaf_findings = []  # Tab 5 -- per-leaf breakdown

if "leaf_tree_summary" not in st.session_state:
    st.session_state.leaf_tree_summary = None  # Tab 3 -- tree-wise summary

if "leaf_tree_id" not in st.session_state:
    st.session_state.leaf_tree_id = None  # which tree the current leaf_findings/images belong to

# Store uploaded images for displaying later
if "disease_images" not in st.session_state:
    st.session_state.disease_images = {}

# Leaf photos, keyed by their ACTUAL filename -- matches Module 01's
# "source_image" field exactly (see api_server.py's _leaf_findings()),
# so lookups are exact instead of guessed by upload position.
if "leaf_images_by_name" not in st.session_state:
    st.session_state.leaf_images_by_name = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_cv_response(data, source: str) -> list[dict]:
    """Accepts several possible Module 1 / 2 response shapes and returns a
    flat list of {"disease", "severity", "stage", "source"} dicts."""
    if isinstance(data, dict) and "findings" in data:
        data = data["findings"]

    out = []
    if isinstance(data, dict):
        # {"leaf_blight": 40, "sooty_mould": 12}
        for disease, val in data.items():
            if isinstance(val, dict):
                sev = val.get("severity", val.get("percentage", 0))
                stage = val.get("stage")
            else:
                sev, stage = val, None
            out.append({"disease": disease, "severity": float(sev), "stage": stage, "source": source})
    elif isinstance(data, list):
        # [{"disease": "leaf_blight", "severity": 40}, ...]
        for row in data:
            disease = row.get("disease") or row.get("name") or row.get("type")
            sev = row.get("severity", row.get("percentage", row.get("pct", 0)))
            stage = row.get("stage")
            if disease is None:
                continue
            out.append({"disease": disease, "severity": float(sev), "stage": stage, "source": source})
    return [f for f in out if f["severity"] and f["severity"] > 0]


def call_cv_module(url: str, api_key: str, files, source: str) -> tuple[list[dict], str | None]:
    """POST uploaded images to a teammate's CV endpoint. Adjust the field
    name ("images") and route if their real API differs."""
    if not url:
        return [], "No endpoint URL set for this module."
    try:
        payload_files = [("images", (f.name, f.getvalue(), f.type)) for f in files]
        headers = {"x-api-key": api_key} if api_key else {}
        resp = requests.post(url.rstrip("/") + "/predict", files=payload_files, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return normalize_cv_response(resp.json(), source), None
    except requests.exceptions.RequestException as e:
        return [], f"Could not reach {source} endpoint: {e}"
    except ValueError as e:
        return [], f"{source} endpoint returned invalid JSON: {e}"


def call_leaf_module(url: str, api_key: str, files, tree_id: str) -> tuple[list[dict], dict | None, list[dict], str | None]:
    """POST leaf photos to Module 1's /predict_full (richer than /predict --
    includes tree_summary for Tab 3 and leaf_findings for Tab 5 alongside the
    usual flat findings list used for the decision engine).
    Returns (findings, tree_summary, leaf_findings, error)."""
    if not url:
        return [], None, [], "No endpoint URL set for this module."
    try:
        payload_files = [("images", (f.name, f.getvalue(), f.type)) for f in files]
        headers = {"x-api-key": api_key} if api_key else {}
        resp = requests.post(
            url.rstrip("/") + "/predict_full", files=payload_files,
            data={"tree_id": tree_id}, headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        findings = normalize_cv_response(data, "leaf (Module 1)")
        tree_summary = data.get("tree_summary")
        leaf_findings = data.get("leaf_findings", [])
        return findings, tree_summary, leaf_findings, None
    except requests.exceptions.RequestException as e:
        return [], None, [], f"Could not reach leaf (Module 1) endpoint: {e}"
    except ValueError as e:
        return [], None, [], f"leaf (Module 1) endpoint returned invalid JSON: {e}"


def merge_findings(new_findings: list[dict]) -> None:
    """Add/replace findings by disease name (last write wins per disease)."""
    current = {f["disease"]: f for f in st.session_state.findings}
    for f in new_findings:
        current[f["disease"]] = f
    st.session_state.findings = list(current.values())


def call_decision_engine(url: str, api_key: str, tree_id: str, findings: list[dict],
                          plantation_age: str, base_grade: str | None) -> tuple[dict | None, str | None]:
    if not url:
        return None, "No decision engine URL set in the sidebar."
    # Module 03's api_server.py accepts leaf_diseases and bark_diseases as
    # two SEPARATE optional dicts (see /generate-recommendation there) --
    # each finding is routed to the right group by its canonical disease
    # key, rather than everything being sent under bark_diseases.
    body = {
        "tree_id": tree_id,
        "leaf_diseases": {},
        "bark_diseases": {},
    }

    for f in findings:
        disease = DISEASE_MAP.get(f["disease"], f["disease"])
        item = {
            "severity_percentage": f["severity"],
            "stage": f.get("stage"),
        }
        if disease in BARK_DISEASES:
            body["bark_diseases"][disease] = item
        elif disease in LEAF_DISEASES:
            body["leaf_diseases"][disease] = item

    try:
        resp = requests.post(
            url.rstrip("/") + "/generate-recommendation",
            json=body,
            headers={"x-api-key": api_key},
            timeout=60,
        )
        if resp.status_code == 401:
            return None, "Decision engine rejected the API key -- check it matches SHARED_SECRET."
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach decision engine: {e}"


# ---------------------------------------------------------------------------
# Sidebar -- endpoint configuration
# ---------------------------------------------------------------------------
with st.sidebar:

    leaf_url = MODULE_01_LEAF_URL
    leaf_key = API_KEY


    decision_url = MODULE_03_DECISION_URL
    decision_key = API_KEY

    st.divider()
    st.markdown("### 🌳 Tree info")
    tree_id = st.text_input("Tree ID", value="tree_001")
    plantation_age = st.selectbox("Plantation age", ["mature", "nursery"])
    base_grade = st.text_input("Usual grade sold (optional)", placeholder="e.g. C-5")

    st.divider()
    if st.button("🔄 Reset demo"):
        st.session_state.findings = []
        st.session_state.report = None
        st.rerun()

st.markdown("""
<div class="axion-hero">
  <h1>🌿 Cinnamon Tree Decision Support System</h1>
  <p>Upload leaf &amp; bark photos → CV models detect disease → decision engine recommends treatment.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 ,tab4, tab5 = st.tabs([
    "1️⃣ Upload & Detect",
        "2️⃣ Bark Image-wise Analysis",
        "3️⃣ Leaf-wise Analysis",
        "4️⃣ Review Findings",
        "5️⃣ Decision Report"])

# ---------------------------------------------------------------------------
# Tab 1 -- upload photos, call Module 1 & Module 2
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🍃 Leaf photos → Module 1")
        leaf_files = st.file_uploader(
            "Upload leaf photos", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="leaf_files",
        )
        if leaf_files:
            st.image([f for f in leaf_files[:5]], width=100)
            if len(leaf_files) > 5:
                st.caption(f"...and {len(leaf_files) - 5} more")
        if st.button("Run Leaf Disease Detection", type="primary", use_container_width=True):
            if not leaf_files:
                st.warning("Upload at least one leaf photo first.")
            else:
                with st.spinner("Module 1 analysing leaf photos..."):
                    found, tree_summary, leaf_findings, err = call_leaf_module(
                        leaf_url, leaf_key, leaf_files, tree_id)
                if err:
                    st.error(err)
                else:
                    # Save uploaded leaf images for Tab 5, keyed by
                    # (tree_id, filename) together -- not filename alone.
                    # Generic filenames like "t2 (1).JPG" repeat across
                    # different trees' photo sets, so filename-only keys
                    # can collide between trees; pairing with tree_id
                    # keeps each tree's photos isolated.
                    for img in leaf_files:
                        st.session_state.leaf_images_by_name[f"{tree_id}::{img.name}"] = img
                    st.session_state.leaf_tree_id = tree_id

                    st.session_state.leaf_tree_summary = tree_summary
                    st.session_state.leaf_findings = leaf_findings
                    if not found:
                        st.info("No leaf disease detected.")
                    else:
                        merge_findings(found)
                        st.success(f"Detected {len(found)} leaf disease finding(s). See tab 4 for the tree-wise summary and tab 3 for the leaf-wise breakdown.")

    with col2:
        st.subheader("🪵 Bark photos → Module 2")
        bark_files = st.file_uploader(
            "Upload bark photos", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="bark_files",
        )
        if bark_files:
            st.image([f for f in bark_files[:5]], width=100)
            if len(bark_files) > 5:
                st.caption(f"...and {len(bark_files) - 5} more")
        if st.button("Run Bark Disease Detection", type="primary", use_container_width=True):
            if not bark_files:
                st.warning("Upload at least one bark photo first.")
            else:
                with st.spinner("Module 2 analysing bark photos..."):
                    from tree_severity_bridge import (
                        BarkSeverityPipeline,
                        load_images_from_streamlit_uploads,
                    )

                    pipeline = get_bark_pipeline()

                images = load_images_from_streamlit_uploads(bark_files)


                # Save uploaded images
                for idx, img in enumerate(bark_files):

                    image_key = f"bark_image_{idx+1}.jpg"

                    st.session_state.disease_images[image_key] = img


                # Run bark analysis
                result = pipeline.analyze(tree_id, images)
               

                # Save image-wise results for Tab 4
                st.session_state.bark_image_results = result.get(
                    "per_image_results",
                    []
                )

                

                found = []

                # Convert pipeline output into DSS format
                for disease, info in result.get("per_disease_bark", {}).items():
                    found.append({
                        "disease": disease,
                        "severity": float(info["pct_bark"]),   # convert fraction → percentage
                        "stage": info.get("stage"),
                        "source": "bark (Module 2)",
                        "image": bark_files[0].name
                    })

                if not found:
                    st.info("No bark disease detected.")
                else:
                    merge_findings(found)
                    st.success(f"Detected {len(found)} bark disease finding(s). See tab 4 for the tree-wise summary and tab 2 for the bark-wise breakdown.")

# ---------------------------------------------------------------------------
# Tab 2 -- Bark Image-wise Analysis
# FIX: this block used to live inside `with tab4:`, after an
# `if not report: st.stop()` guard, so it never rendered until a decision
# report existed. It's now its own top-level tab block, same as
# tab1/tab3/tab4 -- content and logic inside are unchanged.
# ---------------------------------------------------------------------------
with tab2:
    if st.session_state.bark_image_results:
        for result in st.session_state.bark_image_results:
            col1, col2 = st.columns([1, 2])
            with col1:
                image_name = result["image_name"]
                image_file = st.session_state.disease_images.get(image_name)
                if image_file:
                    st.image(image_file, caption=image_name, width=250)
                else:
                    st.warning(f"Image missing: {image_name}")
            with col2:
                disease = result["disease"]
                st.markdown(f"### 🦠 {disease}")
                st.write("Severity:", f"{result['severity_pct']:.2f}%")
                st.write("Confidence:", f"{result['confidence']:.2f}%")
                st.write("QSI:", f"{result['qsi']:.4f}")
                st.markdown(stage_badge_html(result["stage"]), unsafe_allow_html=True)
            st.divider()
    else:
        st.subheader("📸 Bark Image-wise Analysis")
        st.info("Run Bark Disease Detection first.")

# ---------------------------------------------------------------------------
# Tab 4 -- review / edit findings before sending to the decision engine
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Detected findings")

    if st.session_state.findings:
        df = pd.DataFrame(st.session_state.findings)
        df["disease"] = df["disease"].map(lambda d: PRETTY.get(d, d))
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True,
            column_config={
                "severity": st.column_config.NumberColumn("Severity %", min_value=0, max_value=100),
                "disease": st.column_config.SelectboxColumn("Disease", options=list(PRETTY.values())),
                "source": st.column_config.TextColumn("Source", disabled=True),
            },
        )
        # map pretty names back to internal keys before storing
        rev_pretty = {v: k for k, v in PRETTY.items()}
        st.session_state.findings = [
            {**row, "disease": rev_pretty.get(row["disease"], row["disease"])}
            for row in edited.to_dict("records")
            if row.get("disease") 
        ]
    else:
        st.info("No findings yet -- run detection in tab 1, or add one manually below.")

    with st.expander("➕ Add a finding manually"):
        c1, c2, c3 = st.columns(3)
        with c1:
            manual_disease = st.selectbox("Disease", list(PRETTY.values()), key="manual_disease")
        with c2:
            manual_severity = st.slider("Severity %", 0, 100, 30, key="manual_severity")
        with c3:
            manual_stage = st.selectbox(
                "Stage (optional)",
                [None, "Preventive", "Early control", "Active management", "Severe outbreak"],
                key="manual_stage",
            )
        if st.button("Add finding"):
            rev_pretty = {v: k for k, v in PRETTY.items()}
            merge_findings([{
                "disease": rev_pretty[manual_disease],
                "severity": float(manual_severity),
                "stage": manual_stage,
                "source": "manual",
            }])
            st.rerun()


   

# ---------------------------------------------------------------------------
# Tab 4 -- send to the decision engine, render the report
# ---------------------------------------------------------------------------
with tab5:
    # -----------------------------------------------------------------------
    # TAB 4 -- Decision Support output.
    #
    # JSON CONTRACT this tab expects back from the decision engine endpoint
    # (i.e. whatever wraps decision_support.block_report.build(...).render()
    # into the /generate-recommendation response). If the backend response
    # doesn't yet have a field, this tab shows a sensible fallback instead of
    # crashing -- same defensive style as the rest of this file.
    #
    #   detected: { disease_key: {severity_now, stage_now, severity_next,
    #                             stage_next, trend} }
    #   priority_order: [disease_key, ...]      -- bark fixed first, then
    #                    leaf diseases ranked by severity
    #   schedule: { visits: [ {week, day_offset, product, action_ids,
    #                          diseases: [disease_key,...]} ],
    #               notes: [str, ...] }
    #   visit_costs: [ {week, product, diseases:[pretty name,...],
    #                    concentrate, unit, cost, assumed_dose} ]
    #   chem_total, labour_total, block_cost   -- LKR, for the 15-tree block
    #   conflicts / redundancy: [ {message: str, ...} ]
    #   cultural: [ {disease: pretty name, action: str} ]
    #   price_table: [ {grade, week_3, week_12, change, is_major} ]
    #   decision: { options: { treat_and_wait / sell_now / wait_untreated:
    #                          {label, grade, price, revenue, cost, net} },
    #               recommended: key, reason: str }
    #   warnings: [str, ...]                    -- e.g. placeholder data flags
    #
    # NOTE: the original `if not report: st.stop()` guard was replaced with
    # an if/else below, because st.stop() halts the ENTIRE script the moment
    # it runs -- not just this tab -- which was also the cause of Tab 2 never
    # rendering (see the fix note above Tab 2's block). Everything that used
    # to run after that guard is now simply the `else` branch here; no
    # content or computation inside was changed.
    # -----------------------------------------------------------------------

    st.subheader("Generate recommendation")

    if not st.session_state.findings:
        st.warning("No findings to send yet -- add some in tab 1 or tab 2.")
    else:
        st.write(f"Sending **{len(st.session_state.findings)}** finding(s) for tree **{tree_id}**.")

    if st.button("🧠 Generate Decision Report", type="primary", disabled=not st.session_state.findings):
        with st.spinner("Decision engine running the treatment pipeline..."):
            result, err = call_decision_engine(
                decision_url, decision_key, tree_id, st.session_state.findings,
                plantation_age, base_grade,
            )
        if err:
            st.error(err)
        else:
            st.session_state.report = result

    report = st.session_state.report
    if not report:
        st.info("Run the report above to see a recommendation here.")
    else:
        st.divider()

        for w in report.get("warnings", []):
            st.warning(w)

        # local, tab4-only naming helpers -- deliberately NOT touching the
        # PRETTY / DISEASES constants defined at the top of the file, since those
        # belong to the other tabs too.
        def _pretty(d: str) -> str:
            return PRETTY.get(d, d.replace("_", " ").title())

        STAGE_COLOR = {
            "Preventive": "#5C7F4A", "Early control": "#8B7A34",
            "Active management": "#B36A2E", "Severe outbreak": "#8B3A1E",
        }

        def _trend_arrow(t: str) -> str:
            return {"worsening": "🔺", "improving": "🟢", "stable": "➖"}.get(t, "")

        # =========================================================================
        # 1. THE ANSWER, FIRST -- a farmer opening this tab should see the
        #    decision before anything else, with the numbers behind it directly
        #    underneath. Everything else on this tab is supporting detail.
        #
        #    No single "expected grade" is shown or guessed -- the model cannot
        #    honestly promise one. Instead the decision is computed for BOTH C5
        #    and C4, the two grades the great majority of smallholders actually
        #    produce, and shown side by side.
        # =========================================================================
        decision = report.get("decision", {})
        by_grade = decision.get("by_grade", {})
        grades_shown = decision.get("grades", list(by_grade.keys()))

        if by_grade:
            agrees = decision.get("agrees", False)
            banner_bg = "var(--leaf-100)" if agrees else "#FBF2E3"
            banner_border = "var(--leaf-700)" if agrees else "var(--bark-500)"
            st.markdown(f"""
            <div class="axion-card" style="border-left:8px solid {banner_border}; background:{banner_bg};">
              <div style="font-size:13px; color:var(--bark-700); font-weight:600; letter-spacing:.04em; text-transform:uppercase;">
                Recommendation for this tree -- checked against both common grades
              </div>
              <div style="font-size:20px; font-weight:700; color:var(--bark-900); margin:4px 0 8px;">
                {decision.get('summary', '')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(len(grades_shown) or 1)
            for i, g in enumerate(grades_shown):
                d = by_grade.get(g, {})
                rec_key = d.get("recommended")
                opts = d.get("options", {})
                rec = opts.get(rec_key, {})
                with cols[i]:
                    st.markdown(f"**If this batch grades {g}**")
                    st.markdown(f"""
                    <div class="axion-card">
                      <div style="font-weight:700; color:var(--bark-900); font-size:16px;">
                        {rec.get('label', '-')}
                      </div>
                      <div style="margin-top:6px; font-size:14px; color:var(--bark-700);">
                        Revenue: <b>{rec.get('revenue', 0):,.0f} LKR</b><br>
                        Cost: <b>{rec.get('cost', 0):,.0f} LKR</b><br>
                        Net: <b>{rec.get('net', 0):+,.0f} LKR</b>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Compare all options for {g}"):
                        rows = [
                            {"Option": o.get("label", k),
                             "Revenue (LKR)": o.get("revenue", 0),
                             "Cost (LKR)": o.get("cost", 0),
                             "Net (LKR)": o.get("net", 0),
                             "Chosen": "✅" if k == rec_key else ""}
                            for k, o in opts.items()
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if not agrees:
                st.caption("⚠️ The best option isn't the same for C5 and C4 here -- "
                          "worth checking which grade this batch is more likely to be.")
        else:
            st.info("Run the report above to see a recommendation here.")

        st.divider()

        # =========================================================================
        # 2. WHAT'S ON THE TREE -- now vs next week, with the stage each disease
        #    has actually reached (per-disease severity bands).
        # =========================================================================
        detected = report.get("detected", {})
        if detected:
            st.markdown("### 🔬 Your trees, right now")
            cols = st.columns(min(len(detected), 3) or 1)
            for i, (d, v) in enumerate(detected.items()):
                with cols[i % len(cols)]:
                    stage_now = v.get("stage_now", "-")
                    color = STAGE_COLOR.get(stage_now, "#8B5E34")
                    st.markdown(f"""
                    <div class="axion-card">
                      <div style="font-weight:700; font-size:15px; color:var(--bark-900);">{_pretty(d)}</div>
                      <span class="axion-badge" style="background-color:{color};">{stage_now}</span>
                      <div style="margin-top:8px; font-size:14px; color:var(--bark-700);">
                        Now: <b>{v.get('severity_now', 0):.0f}%</b><br>
                        Next week: <b>{v.get('severity_next', 0):.0f}%</b>
                        &nbsp; {_trend_arrow(v.get('trend',''))} {v.get('trend','')}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            priority = report.get("priority_order", [])
            if priority:
                st.caption(
                    "Treatment order (bark first, then leaves by how bad they are): "
                    + " → ".join(_pretty(d) for d in priority)
                )

        # =========================================================================
        # 3. THE TREATMENT PLAN -- calendar of visits, spaced so the same
        #    chemical group isn't used twice too close together, with anything
        #    that could be combined already combined.
        # =========================================================================
        schedule = report.get("schedule", {})
        visits = schedule.get("visits", [])
        visit_costs = report.get("visit_costs", [])

        if visits:
            st.markdown("### 💊 Your treatment plan")
            st.caption("Spray visits are spaced two weeks apart so the same chemical "
                       "isn't relied on too often -- that keeps it working longer.")

            cost_by_week = {vc.get("week"): vc for vc in visit_costs}
            for v in visits:
                wk = v.get("week", 0)
                vc = cost_by_week.get(wk, {})
                when = "Do this now" if wk == 0 else f"In {wk * 2} weeks"
                diseases = ", ".join(_pretty(d) for d in v.get("diseases", []))
                assumed = "  *(estimated dose)*" if vc.get("assumed_dose") else ""

                with st.container():
                    st.markdown(f"""
                    <div class="axion-card">
                      <div style="display:flex; justify-content:space-between; align-items:baseline;">
                        <span style="font-weight:700; color:var(--bark-900); font-size:15px;">🗓️ {when}</span>
                        <span style="color:var(--leaf-700); font-weight:600;">{vc.get('cost', 0):,.0f} LKR</span>
                      </div>
                      <div style="margin-top:4px; font-size:14px;">
                        Spray: <b>{v.get('product', '-').replace('_',' ').title()}</b>{assumed}<br>
                        Treats: {diseases}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # cost roll-up
            m1, m2, m3 = st.columns(3)
            m1.metric("Chemical cost", f"{report.get('chem_total', 0):,.0f} LKR")
            m2.metric("Labour cost", f"{report.get('labour_total', 0):,.0f} LKR")
            m3.metric("Total for this block (15 trees)", f"{report.get('block_cost', 0):,.0f} LKR")
        else:
            st.info("No sprays needed this week -- current stages only call for the "
                   "non-chemical actions below.")

        # safety checks -- shown plainly, not buried
        conflicts = report.get("conflicts", [])
        redundancy = report.get("redundancy", [])
        if conflicts:
            st.warning("⚠️ **Two chemicals that work the same way were both about to be "
                      "used** -- this has been fixed automatically:")
            for c in conflicts:
                st.write(f"- {c.get('message', c)}")
        if redundancy:
            st.success("💰 **Sprays combined to save money:**")
            for r in redundancy:
                st.write(f"- {r.get('message', r)}")

        # cultural (non-chemical) actions
        cultural = report.get("cultural", [])
        if cultural:
            with st.expander(f"🌱 Other things to do -- no chemical, no waiting period ({len(cultural)})"):
                by_disease: dict[str, list[str]] = {}
                for c in cultural:
                    by_disease.setdefault(c.get("disease", "-"), []).append(c.get("action", ""))
                for dis, actions in by_disease.items():
                    st.markdown(f"**{dis}**")
                    for a in actions:
                        st.write(f"- {a}")

        # anything the model considered and rejected
        rejected = report.get("rejected", [])
        if rejected:
            with st.expander("❌ Not doing, and why"):
                for r in rejected:
                    st.write(f"- {r}")

        st.divider()

        # =========================================================================
        # 4. WHAT THE CROP WILL BE WORTH -- all four grades, both horizons,
        #    with the two grades most farmers actually sell at picked out.
        # =========================================================================
        price_table = report.get("price_table", [])
        if price_table:
            st.markdown("### 📈 Forecast prices -- all four grades")
            st.caption("C4 and C5 are highlighted because most farmers sell at these grades.")

            rows = []
            for p in price_table:
                rows.append({
                    "Grade": ("⭐ " if p.get("is_major") else "") + p.get("grade", "-"),
                    "In 3 weeks (LKR/kg)": round(float(p.get("week_3", 0) or 0), 2),
                    "In 12 weeks (LKR/kg)": round(float(p.get("week_12", 0) or 0), 2),
                    "Change": round(float(p.get("change", 0) or 0), 2),
                })
            df_price = pd.DataFrame(rows)

            def _hl(row):
                is_major = row["Grade"].startswith("⭐")
                # st.dataframe's grid renders outside the page's normal DOM,
                # so it can't resolve CSS custom properties like var(--leaf-100)
                # -- that's what was producing the solid black bar. Explicit
                # hex values work correctly here, and the text color is set
                # explicitly too so it can't end up dark-on-dark again.
                style = "background-color: #E4EEDD; color: #243B1E; font-weight:700"
                return [style if is_major else ""] * len(row)

            st.dataframe(
                df_price.style.apply(_hl, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    "In 3 weeks (LKR/kg)": st.column_config.NumberColumn(format="%.2f"),
                    "In 12 weeks (LKR/kg)": st.column_config.NumberColumn(format="%.2f"),
                    "Change": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        with st.expander("📄 Full report (plain text)"):
            st.code(report.get("farmer_report", report.get("summary", "")), language=None)

        sources = report.get("sources", {})
        if sources:
            st.caption(
                "Data sources -- disease: " + sources.get("disease", "?") +
                " | weather: " + sources.get("weather", "?") +
                " | price: " + sources.get("price", "?")
            )

# ---------------------------------------------------------------------------
# Tab 5 -- Leaf-wise Analysis
# FIX: mirrors Tab 2's layout -- photo on the left, findings on the right,
# one row per leaf, instead of a text-only table/cards. Tab 1's leaf
# handler stores each uploaded leaf photo keyed by (tree_id, filename)
# together in st.session_state.leaf_images_by_name -- since Module 01's
# api_server.py echoes back the original filename as "source_image", but
# generic filenames like "t2 (1).JPG" can repeat across different trees'
# photo sets, tree_id is included in the key to keep them from colliding.
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("🍃 Leaf-wise Analysis")
    leaf_findings = st.session_state.leaf_findings
    if leaf_findings:
        current_tree = st.session_state.leaf_tree_id
        for idx, lf in enumerate(leaf_findings):
            source_image = lf.get("source_image")
            image_name = f"{current_tree}::{source_image}" if source_image else None
            image_file = st.session_state.leaf_images_by_name.get(image_name) if image_name else None

            col1, col2 = st.columns([1, 2])
            with col1:
                if image_file:
                    st.image(image_file, caption=source_image, width=250)
                else:
                    st.warning(f"Image missing: {source_image or f'leaf #{idx+1}'}")
            with col2:
                st.markdown(f"### 🍃  {lf.get('disease_name', '-')}")
                st.write("Severity:", f"{lf.get('severity_percentage', 0):.2f}%")
                st.markdown(stage_badge_html(lf.get("stage")), unsafe_allow_html=True)
            st.divider()

        with st.expander("Show as table instead"):
            rows = [{
                "Leaf ID": lf.get("leaf_id"),
                "Disease": lf.get("disease_name"),
                "Severity %": lf.get("severity_percentage"),
                "Stage": lf.get("stage"),
            } for lf in leaf_findings]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run Leaf Disease Detection in tab 1 to see the leaf-wise breakdown here.")