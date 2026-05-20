from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────
# Feature label mapping (for charts)
# ─────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "CreditScore":       "Credit Score",
    "Age":               "Age",
    "Tenure":            "Tenure",
    "Balance":           "Account Balance",
    "NumOfProducts":     "No. of Products",
    "HasCrCard":         "Has Credit Card",
    "IsActiveMember":    "Active Member",
    "EstimatedSalary":   "Est. Salary",
    "Geography_Germany": "From Germany",
    "Geography_Spain":   "From Spain",
    "Gender_Male":       "Gender: Male",
}

RISK_META = {
    "Low Risk":      {"class": "low-risk",       "icon": "🟢", "color": "#22C55E"},
    "Moderate Risk": {"class": "moderate-risk",  "icon": "🟡", "color": "#EAB308"},
    "High Risk":     {"class": "high-risk",      "icon": "🟠", "color": "#F97316"},
    "Critical Risk": {"class": "critical-risk",  "icon": "🔴", "color": "#EF4444"},
}

RETENTION_META = {
    "Low Risk":      ("💚", "Customer appears stable. Continue regular engagement and standard communication cadence."),
    "Moderate Risk": ("👀", "Monitor activity closely and offer personalised loyalty rewards or check-in calls."),
    "High Risk":     ("⚡", "Prioritise direct retention outreach with targeted incentives and dedicated support."),
    "Critical Risk": ("🚨", "Immediate intervention required : escalate to premium support team with personalised offer."),
}


# ─────────────────────────────────────────────────────────
# Load model artefacts
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir   = Path(__file__).resolve().parent
    arts       = base_dir / "artifacts"
    model      = joblib.load(arts / "final_random_forest_churn_model.pkl")
    with open(arts / "model_columns.json") as f:
        columns = json.load(f)
    with open(arts / "model_config.json") as f:
        config  = json.load(f)
    return model, columns, config

model, model_columns, model_config = load_artifacts()
threshold = model_config.get("threshold", 0.4)


# ─────────────────────────────────────────────────────────
# Sidebar — theme toggle + info
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 0.6rem;">
            <div style="font-size:2.2rem;line-height:1;">📊</div>
            <div style="font-size:1.45rem;font-weight:800;margin-top:0.35rem;
                        letter-spacing:-0.03em;">ChurnSense</div>
            <div style="font-size:0.78rem;opacity:0.55;margin-top:0.15rem;">
                Customer Retention Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    theme_choice = st.radio("🎨 Theme", ["Dark", "Light"], horizontal=True, key="app_theme")
    st.divider()

    st.markdown("**📋 How to Use**")
    st.markdown(
        "Fill in the customer profile in the left panel "
        "and click **Predict Churn** to receive:\n"
        "- Churn probability estimate\n"
        "- Risk level classification\n"
        "- Retention recommendation\n"
        "- Feature importance breakdown"
    )
    st.divider()

    st.markdown("**🎯 Risk Levels**")
    st.markdown(
        "🟢 **Low** · < 30%  \n"
        "🟡 **Moderate** · 30 – 50%  \n"
        "🟠 **High** · 50 – 70%  \n"
        "🔴 **Critical** · > 70%"
    )
    st.divider()

    st.markdown("**⚙️ Model Info**")
    st.markdown(
        f"- **Model:** {model_config.get("ML Model Used", "Random Forest")}  \n"
        f"- **Threshold:** {threshold}  \n"
        f"- **Type:** Binary Classification  \n"
        f"- **Features:** {len(model_columns)} variables"
    )


# ─────────────────────────────────────────────────────────
# Theme colour system
# ─────────────────────────────────────────────────────────
if theme_choice == "Light":
    app_bg      = "#F5F0EA"
    card_bg     = "#FFFFFF"
    input_bg    = "#F9F6F2"
    text_main   = "#18181B"
    text_muted  = "#71717A"
    border      = "rgba(249,115,22,0.22)"
    accent      = "#F97316"
    accent_hex6 = "F97316"
    button_bg   = "linear-gradient(135deg,#FB923C,#F97316)"
    button_hov  = "linear-gradient(135deg,#F97316,#EA580C)"
    shadow      = "0 4px 20px rgba(249,115,22,0.10),0 1px 4px rgba(0,0,0,0.05)"
    grid_color  = "rgba(249,115,22,0.12)"
else:
    app_bg      = "#060D1A"
    card_bg     = "#0C1628"
    input_bg    = "#111E35"
    text_main   = "#E2E8F0"
    text_muted  = "#64748B"
    border      = "rgba(59,130,246,0.25)"
    accent      = "#3B82F6"
    accent_hex6 = "3B82F6"
    button_bg   = "linear-gradient(135deg,#1E3A8A,#2563EB)"
    button_hov  = "linear-gradient(135deg,#2563EB,#1D4ED8)"
    shadow      = "0 8px 28px rgba(37,99,235,0.18),0 2px 8px rgba(0,0,0,0.35)"
    grid_color  = "rgba(59,130,246,0.10)"


# ─────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap"
      rel="stylesheet">

<style>
 /* ── Base ── */
  /* 1. Hide ONLY the specific top-right elements (Menu, Deploy) and footer */
  [data-testid="stToolbar"] {{
      display: none !important;
  }}
  footer {{
      display: none !important;
  }}

  /* 2. Make the header background transparent so it blends with your app */
  header {{
      background-color: transparent !important;
  }}

  /* 3. Base app styling */
  html, body, .stApp {{
    background: {app_bg} !important;
    color: {text_main};
    font-family: 'Sora', sans-serif;
  }}
  
  /* 4. Push the main content down slightly so it completely clears the invisible header space */
  .block-container {{
    padding-top: 3.5rem !important; 
    padding-bottom: 2.5rem;
    max-width: 1300px;
  }}
  
  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background: {card_bg} !important;
    border-right: 1px solid {border};
  }}
  section[data-testid="stSidebar"] * {{
    color: {text_main} !important;
  }}

  /* ── Typography ── */
  .app-title {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: {text_main};
    line-height: 1.1;
  }}
  .app-title span {{ color: {accent}; }}
  .app-sub {{
    font-size: 0.9rem;
    color: {text_muted};
    margin-top: 0.3rem;
    margin-bottom: 1.4rem;
  }}
  .section-heading {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {text_main};
    letter-spacing: -0.02em;
    margin-bottom: 0.85rem;
  }}

  /* ── Cards ── */
  /* ── Cards ── */
  .card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: {shadow};
    margin-bottom: 0.85rem;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    text-align: center; /* Added this to center all text inside the cards */
  }}
  .card:hover {{
    transform: translateY(-3px);
    box-shadow: {shadow.replace('0.10', '0.18').replace('0.18', '0.28')};
  }}
  .metric-title {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.28rem;
  }}
  .metric-value {{
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: {text_main};
    letter-spacing: -0.02em;
    line-height: 1.1;
  }}
  .metric-sub {{
    font-size: 0.72rem;
    color: {text_muted};
    margin-top: 0.15rem;
  }}

  /* ── Form container ── */
  div[data-testid="stForm"] {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 1.3rem 1.3rem 1rem;
    box-shadow: {shadow};
  }}

  /* ── Form labels & inputs ── */
  label,
  .stSlider label,
  .stSelectbox label,
  .stRadio label,
  .stNumberInput label {{
    color: {text_main} !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    font-family: 'Sora', sans-serif !important;
  }}
  div[data-baseweb="select"] > div {{
    background-color: {input_bg} !important;
    color: {text_main} !important;
    border-color: {border} !important;
    border-radius: 10px !important;
  }}
  div[data-testid="stNumberInput"] input {{
    background-color: {input_bg} !important;
    color: {text_main} !important;
    border-color: {border} !important;
  }}

  /* ── Submit button ── */
  div[data-testid="stFormSubmitButton"] > button {{
    width: 100%;
    height: 3rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.98rem;
    letter-spacing: 0.02em;
    border: none;
    color: #FFFFFF !important;
    background: {button_bg} !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.22);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    font-family: 'Sora', sans-serif;
  }}
  div[data-testid="stFormSubmitButton"] > button:hover {{
    background: {button_hov} !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 26px rgba(0,0,0,0.32);
    color: #FFFFFF !important;
  }}

  /* ── Risk badges ── */
  .risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 1.05rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    margin: 0.4rem 0 0.9rem;
  }}
  .low-risk     {{ background:rgba(34,197,94,0.15);  color:#22C55E; border:1px solid rgba(34,197,94,0.30); }}
  .moderate-risk{{ background:rgba(234,179,8,0.15);  color:#EAB308; border:1px solid rgba(234,179,8,0.30); }}
  .high-risk    {{ background:rgba(249,115,22,0.15); color:#F97316; border:1px solid rgba(249,115,22,0.30); }}
  .critical-risk{{ background:rgba(239,68,68,0.15);  color:#EF4444; border:1px solid rgba(239,68,68,0.30); }}

  /* ── Retention box ── */
  .retention-box {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 1rem 1.15rem;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    box-shadow: {shadow};
    margin-bottom: 0.9rem;
  }}
  .ret-icon   {{ font-size:1.45rem; line-height:1; padding-top:0.1rem; }}
  .ret-label  {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {accent};
    margin-bottom: 0.2rem;
  }}
  .ret-text   {{ font-size:0.9rem; color:{text_main}; line-height:1.55; }}

  /* ── Empty state ── */
  .empty-state {{
    background: {card_bg};
    border: 1px dashed {border};
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    box-shadow: {shadow};
  }}
  .empty-icon  {{ font-size:2.6rem; margin-bottom:0.85rem; }}
  .empty-title {{ font-size:1.1rem; font-weight:700; color:{text_main}; margin-bottom:0.4rem; }}
  .empty-desc  {{ font-size:0.875rem; color:{text_muted}; line-height:1.65; }}
  .empty-desc b{{ color:{text_main}; }}

  /* ── Misc ── */
  hr {{ border-color:{border}; }}
  div[data-testid="stAlert"]   {{ border-radius:12px !important; }}
  div[data-testid="stExpander"]{{ border-radius:12px !important; }}
  div[data-testid="stPlotlyChart"] {{ border-radius:12px; overflow:hidden; }}

  /* ── Gauge threshold label ── */
  .gauge-caption {{
    text-align:center;
    font-size:0.78rem;
    color:{text_muted};
    margin-top:-0.4rem;
    margin-bottom:0.6rem;
    font-family:'DM Mono', monospace;
  }}

  /* ── Staggered entrance ── */
  @keyframes fadeSlideUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0);    }}
  }}
  .a1 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.05s both; }}
  .a2 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.12s both; }}
  .a3 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.19s both; }}
  .a4 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.26s both; }}
  .a5 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.33s both; }}
  .a6 {{ animation: fadeSlideUp 0.40s cubic-bezier(.22,1,.36,1) 0.40s both; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob < 0.30: return "Low Risk"
    if prob < 0.50: return "Moderate Risk"
    if prob < 0.70: return "High Risk"
    return "Critical Risk"


def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([{
        "CreditScore":       data["CreditScore"],
        "Age":               data["Age"],
        "Tenure":            data["Tenure"],
        "Balance":           data["Balance"],
        "NumOfProducts":     data["NumOfProducts"],
        "HasCrCard":         data["HasCrCard"],
        "IsActiveMember":    data["IsActiveMember"],
        "EstimatedSalary":   data["EstimatedSalary"],
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "Geography_Spain":   1 if data["Geography"] == "Spain"   else 0,
        "Gender_Male":       1 if data["Gender"]    == "Male"    else 0,
    }])
    return df.reindex(columns=model_columns, fill_value=0)


def build_gauge(prob_pct: float, risk_color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={
            "suffix": "%",
            "font": {"size": 46, "family": "DM Mono, monospace", "color": text_main},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": text_muted,
                "tickfont": {"color": text_muted, "size": 11, "family": "Sora, sans-serif"},
                "nticks": 6,
            },
            "bar":       {"color": risk_color, "thickness": 0.24},
            "bgcolor":   "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(34,197,94,0.10)"},
                {"range": [30, 50], "color": "rgba(234,179,8,0.10)"},
                {"range": [50, 70], "color": "rgba(249,115,22,0.10)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.10)"},
            ],
            "threshold": {
                "line":      {"color": accent, "width": 3},
                "thickness": 0.80,
                "value":     threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=16, b=0),
        font={"family": "Sora, sans-serif", "color": text_main},
    )
    return fig


def build_importance_chart(mdl, columns: list) -> go.Figure:
    importances = mdl.feature_importances_
    labels = [FEATURE_LABELS.get(c, c) for c in columns]
    df = (
        pd.DataFrame({"Feature": labels, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(8)
    )
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker={
    "color": df["Importance"],
    "colorscale": [
        [0, f"rgba({int(accent_hex6[0:2], 16)}, {int(accent_hex6[2:4], 16)}, {int(accent_hex6[4:6], 16)}, 0.20)"],
        [1, f"rgb({int(accent_hex6[0:2], 16)}, {int(accent_hex6[2:4], 16)}, {int(accent_hex6[4:6], 16)})"]
    ],
    "line": {"width": 0},
},
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        textfont={"color": text_muted, "size": 11, "family": "DM Mono, monospace"},
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=65, t=8, b=8),
        xaxis=dict(
            showgrid=True, gridcolor=grid_color, zeroline=False,
            showticklabels=False, fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont={"color": text_main, "size": 12, "family": "Sora, sans-serif"},
            fixedrange=True,
        ),
        font={"family": "Sora, sans-serif", "color": text_main},
        hoverlabel=dict(
            bgcolor=card_bg, bordercolor=border,
            font_family="Sora, sans-serif", font_color=text_main,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-title">Churn<span>Sense</span></div>'
    '<div class="app-sub">Customer churn intelligence · Powered by Random Forest</div>',
    unsafe_allow_html=True,
)

# Top stat strip
tc1, tc2, tc3, tc4 = st.columns(4)

stats = [
    {
        "title": "Prediction Model",
        "value": "Random Forest",
        "sub": "trained classifier"
    },
    {
        "title": "Decision Threshold",
        "value": f"{int(threshold * 100)}%",
        "sub": "churn cutoff"
    },
    {
        "title": "Classification",
        "value": "Binary",
        "sub": "churn / no churn"
    },
    {
        "title": "Input Features",
        "value": str(len(model_columns)),
        "sub": "customer variables"
    },
]

for col, item in zip([tc1, tc2, tc3, tc4], stats):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">{item["title"]}</div>
                <div class="metric-value" style="font-size:1.1rem;">{item["value"]}</div>
                <div class="metric-sub">{item["sub"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.2], gap="large")


# ── LEFT: input form ──────────────────────────────────────
with left_col:
    st.markdown('<div class="section-heading">Customer Profile</div>', unsafe_allow_html=True)

    with st.form("customer_form"):

        credit_score = st.slider("Credit Score", 300, 900, 650, 10,
                                 help="FICO-style credit score (300 = poor, 900 = excellent)")

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender    = st.radio("Gender", ["Female", "Male"], horizontal=True)

        a_col, t_col = st.columns(2)
        with a_col:
            age    = st.slider("Age", 18, 100, 40, 1)
        with t_col:
            tenure = st.slider("Tenure (years)", 0, 10, 3, 1)

        balance          = st.slider("Account Balance",  0, 250_000, 80_000, 500)
        estimated_salary = st.slider("Estimated Salary", 0, 200_000, 100_000, 500)

        p_col, c_col, ac_col = st.columns(3)
        with p_col:
            num_products = st.selectbox("Products", [1, 2, 3, 4],
                                        help="Number of bank products held")
        with c_col:
            has_cr_card  = st.radio("Credit Card?", ["Yes", "No"])
        with ac_col:
            is_active    = st.radio("Active?", ["Yes", "No"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Predict Churn")


# ── RIGHT: prediction output ──────────────────────────────
with right_col:
    st.markdown('<div class="section-heading">Prediction Output</div>', unsafe_allow_html=True)

    if submitted:
        # ── Build input & predict ──
        customer_data = {
            "CreditScore":    credit_score,
            "Geography":      geography,
            "Gender":         gender,
            "Age":            age,
            "Tenure":         tenure,
            "Balance":        float(balance),
            "NumOfProducts":  num_products,
            "HasCrCard":      1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active   == "Yes" else 0,
            "EstimatedSalary": float(estimated_salary),
        }
        processed   = preprocess_input(customer_data)
        churn_prob  = float(model.predict_proba(processed)[0][1])
        churn_pct   = round(churn_prob * 100, 2)
        prediction  = "Churn" if churn_prob >= threshold else "No Churn"
        risk        = get_risk_level(churn_prob)
        risk_color  = RISK_META[risk]["color"]
        risk_icon   = RISK_META[risk]["icon"]
        risk_class  = RISK_META[risk]["class"]
        ret_icon, ret_text = RETENTION_META[risk]
        pred_color  = "#EF4444" if prediction == "Churn" else "#22C55E"
        delta_from_threshold = churn_pct - threshold * 100

        # ── Metric cards ──
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
                <div class="card a1">
                    <div class="metric-title">Churn Probability</div>
                    <div class="metric-value" style="color:{risk_color}">{churn_pct}%</div>
                    <div class="metric-sub">model estimate</div>
                </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
                <div class="card a2">
                    <div class="metric-title">Verdict</div>
                    <div class="metric-value" style="font-size:1.2rem;color:{pred_color}">
                        {prediction}
                    </div>
                    <div class="metric-sub">at threshold {int(threshold*100)}%</div>
                </div>""", unsafe_allow_html=True)
        with mc3:
            delta_sign  = "+" if delta_from_threshold >= 0 else ""
            delta_color = "#EF4444" if delta_from_threshold >= 0 else "#22C55E"
            st.markdown(f"""
                <div class="card a3">
                    <div class="metric-title">vs. Threshold</div>
                    <div class="metric-value" style="font-size:1.2rem;color:{delta_color}">
                        {delta_sign}{round(delta_from_threshold, 2)}%
                    </div>
                    <div class="metric-sub">margin from cutoff</div>
                </div>""", unsafe_allow_html=True)

        # ── Risk badge ──
        st.markdown(
            f'<div class="risk-badge {risk_class} a4">{risk_icon} {risk}</div>',
            unsafe_allow_html=True,
        )

        # ── Gauge ──
        st.plotly_chart(
            build_gauge(churn_pct, risk_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f'<div class="gauge-caption">Vertical marker at <b>{int(threshold*100)}%</b> '
            f'= decision threshold</div>',
            unsafe_allow_html=True,
        )

        # ── Retention recommendation ──
        st.markdown(f"""
            <div class="retention-box a5">
                <div class="ret-icon">{ret_icon}</div>
                <div>
                    <div class="ret-label">Retention Recommendation</div>
                    <div class="ret-text">{ret_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Feature importance ──
        with st.expander("📊 Feature Importance (model-level)"):
            st.caption("Top 8 features by mean impurity decrease — not customer-specific.")
            st.plotly_chart(
                build_importance_chart(model, model_columns),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Customer summary ──
        with st.expander("📋 Customer Input Summary"):
            readable = {
                "Credit Score":    customer_data["CreditScore"],
                "Geography":       customer_data["Geography"],
                "Gender":          customer_data["Gender"],
                "Age":             customer_data["Age"],
                "Tenure (yrs)":    customer_data["Tenure"],
                "Balance":         f"{customer_data['Balance']:,.0f}",
                "Products":        customer_data["NumOfProducts"],
                "Has Credit Card": "Yes" if customer_data["HasCrCard"] else "No",
                "Active Member":   "Yes" if customer_data["IsActiveMember"] else "No",
                "Est. Salary":     f"{customer_data['EstimatedSalary']:,.0f}",
            }
            st.dataframe(
                pd.DataFrame.from_dict(readable, orient="index", columns=["Value"]),
                use_container_width=True,
            )

    else:
        # ── Empty state ──
        st.markdown(
            f"""<div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <div class="empty-title">Ready to predict</div>
                    <div class="empty-desc">
                        Fill in the customer profile on the left<br>
                        and click <b>Predict Churn</b> to see the results.
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )