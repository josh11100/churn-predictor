import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

*, body { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #13151f !important;
    border-color: #252836 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
label { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 500 !important;
        letter-spacing: 0.06em !important; text-transform: uppercase !important; }

.card { background: #13151f; border: 1px solid #1e2130; border-radius: 14px; padding: 1.4rem 1.6rem; }
.card-accent { background: linear-gradient(135deg, #0d1117 0%, #13151f 100%);
               border: 1px solid #1e2130; border-radius: 14px; padding: 1.8rem 2rem; }
.section-label { font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
                 letter-spacing: 0.18em; text-transform: uppercase; color: #475569;
                 margin-bottom: 1rem; margin-top: 0.2rem; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; }
.metric-sub { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
.pill { display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; }
.pill-high   { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.pill-medium { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.pill-low    { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.divider { border: none; border-top: 1px solid #1e2130; margin: 1.4rem 0; }
.prob-bar-bg   { background: #1e2130; border-radius: 999px; height: 8px; width: 100%; margin-top: 10px; }
.prob-bar-fill { height: 8px; border-radius: 999px; }
.action-box { border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.6; margin-top: 0.5rem; }
.action-high   { background: rgba(239,68,68,0.08);  border-left: 3px solid #f87171; color: #fca5a5; }
.action-medium { background: rgba(245,158,11,0.08); border-left: 3px solid #fbbf24; color: #fde68a; }
.action-low    { background: rgba(34,197,94,0.08);  border-left: 3px solid #4ade80; color: #86efac; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

artifact = load_model()
model    = artifact["model"]
features = artifact["features"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <span style="font-family: Syne, sans-serif; font-size: 1.6rem; font-weight: 800; color: #f1f5f9;">
        📉 ChurnIQ
    </span>
    <span style="margin-left: 12px; font-size: 0.82rem; color: #475569;">
        Customer Retention Intelligence
    </span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑 Single Customer", "📂 Bulk CSV Upload"])

# ═══════════════════════════════════════════════════
# TAB 1 — Single customer
# ═══════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div class="card-accent">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender     = st.selectbox("Gender", ["Male", "Female"])
            partner    = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Account</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c4:
            monthly  = st.number_input("Monthly Charges ($)", 10.0, 120.0, 65.0, step=0.5)
            payment  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_bk      = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        with c6:
            device_prot   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support  = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv  = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        total = monthly * tenure
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Analyze Churn Risk →", use_container_width=True, type="primary")

    def build_input():
        yn = lambda v: 1 if v == "Yes" else 0
        addons     = [online_sec, online_bk, device_prot, tech_support, streaming_tv, streaming_mov]
        num_addons = sum(1 for a in addons if a == "Yes")
        row = {
            "gender":           0 if gender == "Female" else 1,
            "SeniorCitizen":    yn(senior),
            "Partner":          yn(partner),
            "Dependents":       yn(dependents),
            "tenure":           tenure,
            "PhoneService":     yn(phone_service),
            "MultipleLines":    yn(multiple_lines),
            "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
            "OnlineSecurity":   yn(online_sec),
            "OnlineBackup":     yn(online_bk),
            "DeviceProtection": yn(device_prot),
            "TechSupport":      yn(tech_support),
            "StreamingTV":      yn(streaming_tv),
            "StreamingMovies":  yn(streaming_mov),
            "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
            "PaperlessBilling": yn(paperless),
            "PaymentMethod":    {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
                                 "Electronic check": 2, "Mailed check": 3}[payment],
            "MonthlyCharges":   monthly,
            "TotalCharges":     total,
            "charges_per_month": total / (tenure + 1),
            "is_new_customer":  1 if tenure <= 3 else 0,
            "num_addons":       num_addons,
        }
        return pd.DataFrame([row])[features]

    with right:
        if predict_btn:
            X_input = build_input()
            proba   = model.predict_proba(X_input)[0][1]
            pct     = proba * 100
            if pct >= 65:   risk_label, risk_class, bar_color = "High Risk",   "high",   "#f87171"
            elif pct >= 35: risk_label, risk_class, bar_color = "Medium Risk", "medium", "#fbbf24"
            else:           risk_label, risk_class, bar_color = "Low Risk",    "low",    "#4ade80"

            st.markdown(f"""
            <div class="card" style="margin-bottom: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="metric-val" style="color:{bar_color};">{pct:.1f}%</div>
                        <div class="metric-sub">Churn Probability</div>
                    </div>
                    <span class="pill pill-{risk_class}">{risk_label}</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val">${monthly:.0f}</div><div class="metric-sub">Monthly at Risk</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val">${monthly*12:.0f}</div><div class="metric-sub">Annual at Risk</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val">{tenure}mo</div><div class="metric-sub">Tenure</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Recommended Action</div>', unsafe_allow_html=True)
            actions = {
                "high":   "🔴 <strong>Intervene immediately.</strong> Offer a discount, loyalty reward, or personal outreach.",
                "medium": "🟡 <strong>Monitor closely.</strong> Send a proactive check-in or highlight unused features.",
                "low":    "🟢 <strong>No action needed.</strong> This customer appears satisfied. Consider an upsell offer."
            }
            st.markdown(f'<div class="action-box action-{risk_class}">{actions[risk_class]}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="section-label">Why this prediction?</div>', unsafe_allow_html=True)
            st.caption("Features pushing churn risk up (red) or down (blue)")
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            fig, ax = plt.subplots(figsize=(8, 3.8))
            fig.patch.set_facecolor("#13151f")
            ax.set_facecolor("#13151f")
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0], base_values=explainer.expected_value,
                data=X_input.iloc[0].values, feature_names=features),
                show=False, max_display=8)
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color("#94a3b8"); text.set_fontsize(8)
            ax.spines[:].set_color("#1e2130")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown("""
            <div class="card-accent" style="text-align:center; padding:3.5rem 2rem; margin-top:1rem;">
                <div style="font-size:3rem; margin-bottom:1rem;">📉</div>
                <div style="font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#f1f5f9; margin-bottom:0.5rem;">Ready to analyze</div>
                <div style="color:#475569; font-size:0.88rem; line-height:1.7; max-width:300px; margin:0 auto;">
                    Fill in a customer profile on the left and click <strong style="color:#94a3b8;">Analyze Churn Risk →</strong>
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            with p1: st.markdown('<div class="card" style="text-align:center;"><div class="metric-val">0.847</div><div class="metric-sub">ROC-AUC</div></div>', unsafe_allow_html=True)
            with p2: st.markdown('<div class="card" style="text-align:center;"><div class="metric-val">79%</div><div class="metric-sub">Churn Recall</div></div>', unsafe_allow_html=True)
            with p3: st.markdown('<div class="card" style="text-align:center;"><div class="metric-val">7K</div><div class="metric-sub">Training Rows</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# TAB 2 — Bulk CSV upload
# ═══════════════════════════════════════════════════
with tab2:

    def preprocess_uploaded(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "customerID" in df.columns: df = df.drop(columns=["customerID"])
        if "Churn"      in df.columns: df = df.drop(columns=["Churn"])
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
        df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["is_new_customer"]   = (df["tenure"] <= 3).astype(int)
        addons = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                  "TechSupport","StreamingTV","StreamingMovies"]
        df["num_addons"] = df[addons].apply(lambda r: sum(v=="Yes" for v in r), axis=1)
        binary_cols = ["Partner","Dependents","PhoneService","PaperlessBilling",
                       "MultipleLines","OnlineSecurity","OnlineBackup",
                       "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        for col in binary_cols:
            df[col] = df[col].map({"Yes":1,"No":0,"No phone service":0,"No internet service":0})
        df["gender"]          = df["gender"].map({"Female":0,"Male":1})
        df["InternetService"] = df["InternetService"].map({"DSL":0,"Fiber optic":1,"No":2})
        df["Contract"]        = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})
        df["PaymentMethod"]   = df["PaymentMethod"].map({
            "Bank transfer (automatic)":0,"Credit card (automatic)":1,
            "Electronic check":2,"Mailed check":3})
        return df[features]

    st.markdown("### 📂 Bulk Churn Analysis")
    st.markdown("Upload a CSV of your customers and get churn predictions for every row instantly.")

    col_info, col_upload = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown('<div class="section-label">Required CSV columns</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card" style="font-size:0.82rem; line-height:2.2; color:#94a3b8;">
            customerID &nbsp;·&nbsp; gender &nbsp;·&nbsp; SeniorCitizen<br>
            Partner &nbsp;·&nbsp; Dependents &nbsp;·&nbsp; tenure<br>
            PhoneService &nbsp;·&nbsp; MultipleLines<br>
            InternetService &nbsp;·&nbsp; OnlineSecurity<br>
            OnlineBackup &nbsp;·&nbsp; DeviceProtection<br>
            TechSupport &nbsp;·&nbsp; StreamingTV &nbsp;·&nbsp; StreamingMovies<br>
            Contract &nbsp;·&nbsp; PaperlessBilling<br>
            PaymentMethod &nbsp;·&nbsp; MonthlyCharges &nbsp;·&nbsp; TotalCharges
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        sample     = pd.read_csv("data/telco_churn.csv").head(10)
        csv_sample = sample.to_csv(index=False).encode()
        st.download_button("⬇️ Download sample CSV", data=csv_sample,
                           file_name="sample_customers.csv", mime="text/csv",
                           use_container_width=True)

    with col_upload:
        uploaded = st.file_uploader("Upload your customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df       = pd.read_csv(uploaded)
            customer_ids = raw_df["customerID"].values if "customerID" in raw_df.columns \
                           else [f"Customer {i+1}" for i in range(len(raw_df))]
            X_bulk  = preprocess_uploaded(raw_df)
            probas  = model.predict_proba(X_bulk)[:, 1]

            def rlabel(p):
                if p >= 0.65: return "🔴 High"
                if p >= 0.35: return "🟡 Medium"
                return "🟢 Low"

            results = pd.DataFrame({
                "Customer ID":       customer_ids,
                "Churn Probability": [f"{p*100:.1f}%" for p in probas],
                "Risk Level":        [rlabel(p) for p in probas],
                "Monthly Charges":   raw_df["MonthlyCharges"].values if "MonthlyCharges" in raw_df.columns else ["—"]*len(raw_df),
                "Contract":          raw_df["Contract"].values if "Contract" in raw_df.columns else ["—"]*len(raw_df),
            })

            high   = sum(p >= 0.65 for p in probas)
            medium = sum(0.35 <= p < 0.65 for p in probas)
            low    = sum(p < 0.35 for p in probas)
            at_risk_rev = float(raw_df["MonthlyCharges"][probas >= 0.65].sum()) if "MonthlyCharges" in raw_df.columns else 0

            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val" style="color:#f87171;">{high}</div><div class="metric-sub">High Risk</div></div>', unsafe_allow_html=True)
            with s2: st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val" style="color:#fbbf24;">{medium}</div><div class="metric-sub">Medium Risk</div></div>', unsafe_allow_html=True)
            with s3: st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val" style="color:#4ade80;">{low}</div><div class="metric-sub">Low Risk</div></div>', unsafe_allow_html=True)
            with s4: st.markdown(f'<div class="card" style="text-align:center;"><div class="metric-val">${at_risk_rev:,.0f}</div><div class="metric-sub">Monthly $ at Risk</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Results — sorted by churn risk</div>', unsafe_allow_html=True)
            results["_prob"] = probas
            results = results.sort_values("_prob", ascending=False).drop(columns=["_prob"])
            st.dataframe(results, use_container_width=True, hide_index=True)

            csv_out = results.to_csv(index=False).encode()
            st.download_button("⬇️ Download results as CSV", data=csv_out,
                               file_name="churn_predictions.csv", mime="text/csv",
                               use_container_width=True, type="primary")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.caption("Make sure your CSV has all the required columns listed above.")
    else:
        st.markdown("""
        <div class="card-accent" style="text-align:center; padding:3rem 2rem; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📂</div>
            <div style="font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-bottom:0.5rem;">
                Upload a customer list
            </div>
            <div style="color:#475569; font-size:0.85rem; line-height:1.7; max-width:320px; margin:0 auto;">
                Drop in a CSV export of your customers and get churn predictions for everyone at once.
                Download the sample CSV on the left to see the expected format.
            </div>
        </div>""", unsafe_allow_html=True)
