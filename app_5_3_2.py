import pickle
import os
import pandas as pd
import numpy as np

# ── binary_encode — MUST be defined before pickle.load ───────
def binary_encode(x):
    return (pd.DataFrame(x) == 'Yes').astype(int).values

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import lime
import lime.lime_tabular
import dice_ml
from dice_ml import Dice
import warnings
import io
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# ── PAGE CONFIG
st.set_page_config(
    page_title="XAI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"   # sidebar default band
)

# ── PREMIUM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e0e0e0;}
[data-testid="stSidebar"]{background:rgba(255,255,255,0.05);backdrop-filter:blur(20px);border-right:1px solid rgba(255,255,255,0.1);}
.glass{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.mcard{background:linear-gradient(135deg,rgba(78,205,196,0.15),rgba(85,98,212,0.15));border:1px solid rgba(78,205,196,0.3);border-radius:12px;padding:16px;text-align:center;margin:4px;}
.mval{font-size:1.9rem;font-weight:700;color:#4ecdc4;}
.mlbl{font-size:0.75rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;}
.htitle{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#4ecdc4,#556bd4,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:6px;}
.hsub{text-align:center;color:#aaa;font-size:1rem;margin-bottom:24px;}
.approved{background:linear-gradient(135deg,#11998e,#38ef7d);color:white;padding:14px 28px;border-radius:50px;font-size:1.4rem;font-weight:700;text-align:center;box-shadow:0 4px 20px rgba(56,239,125,0.4);}
.denied{background:linear-gradient(135deg,#c0392b,#e74c3c);color:white;padding:14px 28px;border-radius:50px;font-size:1.4rem;font-weight:700;text-align:center;box-shadow:0 4px 20px rgba(231,76,60,0.4);}
.sh{font-size:1.2rem;font-weight:700;color:#4ecdc4;border-left:4px solid #4ecdc4;padding-left:10px;margin:14px 0 10px 0;}
.ibox{background:rgba(78,205,196,0.1);border:1px solid rgba(78,205,196,0.3);border-radius:10px;padding:10px 14px;margin:6px 0;font-size:0.88rem;}
.pbar{background:rgba(255,255,255,0.1);border-radius:50px;height:12px;margin:6px 0;overflow:hidden;}
.pbar-g{background:linear-gradient(90deg,#11998e,#38ef7d);height:100%;border-radius:50px;}
.pbar-r{background:linear-gradient(90deg,#c0392b,#e74c3c);height:100%;border-radius:50px;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.05);border-radius:12px;padding:4px;}
.stTabs [aria-selected="true"]{background:rgba(78,205,196,0.2)!important;color:#4ecdc4!important;}

/* ── NAV BUTTONS ── */
.nav-btn > div > button {
    background: rgba(255,255,255,0.06) !important;
    color: #bbb !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    padding: 8px 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    box-shadow: none !important;
}
.nav-btn > div > button:hover {
    background: rgba(78,205,196,0.15) !important;
    color: #4ecdc4 !important;
    border-color: rgba(78,205,196,0.5) !important;
}
.nav-active > div > button {
    background: linear-gradient(135deg,#4ecdc4,#556bd4) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(78,205,196,0.35) !important;
}

/* Predict page submit button */
.stButton>button{background:linear-gradient(135deg,#4ecdc4,#556bd4);color:white;border:none;border-radius:10px;padding:12px 24px;font-size:1rem;font-weight:600;width:100%;box-shadow:0 4px 15px rgba(78,205,196,0.3);}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pkl_path = 'model/xai_model.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    st.error("❌ model/xai_model.pkl not found! Notebook pehle run karo.")
    st.stop()


# ── SIDEBAR (info only) ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-size:2.5rem;'>🏦</div>
        <div style='font-size:1.2rem;font-weight:700;color:#4ecdc4;'>XAI Loan System</div>
        <div style='font-size:0.75rem;color:#888;margin-top:4px;'>Explainable AI Platform</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class='ibox'>
        <b style='color:#4ecdc4;'>XAI Toolkit</b><br><br>
        🔷 <b>SHAP</b> — Feature attribution<br>
        🟡 <b>LIME</b> — Local explanation<br>
        🔵 <b>DiCE</b> — Counterfactuals
    </div>
    <div style='text-align:center;margin-top:16px;font-size:0.73rem;color:#666;'>
        Built by <span style='color:#4ecdc4;font-weight:600;'>Yash Shakya</span><br>
        B.Tech CSE | IIT Guwahati AI/ML
    </div>""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("⚡ Loading model..."):
    d = load_model()

rf            = d['rf_sklearn']
pipe          = d['pipe']
preprocessing = d['preprocessing']
x_train_np    = d['x_train_np']
x_test_np     = d['x_test_np']
x_train       = d['x_train']
x_test        = d['x_test']
y_train       = d['y_train']
y_test        = d['y_test']
feat_names    = d['feat_names']
feat_names    = list(feat_names)  # ensure list
metrics       = d['metrics']
ds            = d.get('ds', pd.DataFrame())


# ── TOP NAVIGATION (session_state — always visible) ───────────
pages = [
    "🏠 Dashboard",
    "🔮 Predict & Explain",
    "📊 SHAP Analysis",
    "🟡 LIME Explanation",
    "🔵 DiCE Counterfactuals"
]

if 'page' not in st.session_state:
    st.session_state.page = pages[0]

nav_cols = st.columns(5)
for i, (col, p) in enumerate(zip(nav_cols, pages)):
    with col:
        css_class = "nav-active" if st.session_state.page == p else "nav-btn"
        st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
        if st.button(p, key=f"nav_{i}"):
            st.session_state.page = p
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid rgba(255,255,255,0.08);margin:8px 0 20px 0;'>", unsafe_allow_html=True)

page = st.session_state.page


# ── HELPER FUNCTIONS ──────────────────────────────────────────
def pbar(val, color):
    cls = 'pbar-g' if color == 'green' else 'pbar-r'
    label = '✅ Approval' if color == 'green' else '❌ Denial'
    hex_c = '#2ecc71' if color == 'green' else '#e74c3c'
    return f"""
    <div style='margin-bottom:12px;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
            <span style='color:{hex_c};font-weight:600;'>{label}</span>
            <span style='color:{hex_c};font-weight:700;'>{val*100:.1f}%</span>
        </div>
        <div class='pbar'><div class='{cls}' style='width:{val*100:.1f}%'></div></div>
    </div>"""

def plotly_hbar(xvals, yvals, colors, title="", height=400):
    fig = go.Figure(go.Bar(
        x=xvals, y=yvals, orientation='h', marker_color=colors,
        text=[f"{v:+.3f}" for v in xvals], textposition='outside'
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=height,
        margin=dict(t=50 if title else 10, b=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("<div class='htitle'>🏦 XAI Loan Approval System</div>", unsafe_allow_html=True)
    st.markdown("<div class='hsub'>Explainable AI — SHAP • LIME • DiCE • Random Forest from Scratch</div>", unsafe_allow_html=True)

    icons = ["🎯","🔬","📡","⚡","📈"]
    cols  = st.columns(5)
    for col, (name, val), icon in zip(cols, metrics.items(), icons):
        with col:
            st.markdown(f"<div class='mcard'><div style='font-size:1.4rem'>{icon}</div><div class='mval'>{val:.3f}</div><div class='mlbl'>{name}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not ds.empty and 'loan_status' in ds.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sh'>📊 Loan Status Distribution</div>", unsafe_allow_html=True)
            counts = ds['loan_status'].value_counts()
            fig = go.Figure(go.Pie(
                labels=['Denied','Approved'],
                values=[counts.get(0,0), counts.get(1,0)],
                hole=0.55, marker_colors=['#e74c3c','#2ecc71']
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                              height=280, margin=dict(t=10,b=10),
                              legend=dict(font=dict(color='white')))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sh'>💳 Credit Score Distribution</div>", unsafe_allow_html=True)
            fig2 = go.Figure()
            if 'credit_score' in ds.columns:
                fig2.add_trace(go.Histogram(x=ds[ds['loan_status']==0]['credit_score'],
                                            name='Denied', marker_color='#e74c3c', opacity=0.7, nbinsx=30))
                fig2.add_trace(go.Histogram(x=ds[ds['loan_status']==1]['credit_score'],
                                            name='Approved', marker_color='#2ecc71', opacity=0.7, nbinsx=30))
            fig2.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                               height=280, margin=dict(t=10,b=10),
                               xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                               yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>📋 Dataset Overview</div>", unsafe_allow_html=True)
    if not ds.empty:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Samples", f"{len(ds):,}")
        c2.metric("Features", f"{len(ds.columns)-1}")
        c3.metric("Approved", f"{(ds['loan_status']==1).sum():,}" if 'loan_status' in ds.columns else "N/A")
        c4.metric("Denied",   f"{(ds['loan_status']==0).sum():,}" if 'loan_status' in ds.columns else "N/A")
        st.dataframe(ds.head(8), use_container_width=True, height=220)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict & Explain":
    st.markdown("<div class='htitle'>🔮 Loan Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='hsub'>Enter applicant details — instant AI prediction with explanation</div>", unsafe_allow_html=True)

    with st.form("pred_form"):
        st.markdown("<div class='sh'>👤 Applicant Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age          = st.slider("Age", 18, 80, 30)
            income       = st.number_input("Annual Income (₹)", 10000, 10000000, 60000, step=5000)
            emp_exp      = st.slider("Employment Experience (yrs)", 0, 30, 3)
        with c2:
            loan_amnt    = st.number_input("Loan Amount (₹)", 500, 10000000, 10000, step=500)
            loan_int     = st.slider("Interest Rate (%)", 1.0, 25.0, 12.0, step=0.5)
            loan_pct     = st.slider("Loan % of Income", 0.01, 0.90, 0.20, step=0.01)
        with c3:
            cred_hist    = st.slider("Credit History (yrs)", 1, 30, 5)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            home_own     = st.selectbox("Home Ownership", ['RENT','OWN','MORTGAGE','OTHER'])

        c4, c5 = st.columns(2)
        with c4:
            loan_intent  = st.selectbox("Loan Intent", ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION'])
        with c5:
            prev_default = st.selectbox("Previous Default?", ['No','Yes'])

        submitted = st.form_submit_button("🚀 Predict & Explain")

    if submitted:
        input_df = pd.DataFrame([{
            'person_age': age, 'person_income': income,
            'person_emp_exp': emp_exp, 'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int, 'loan_percent_income': loan_pct,
            'cb_person_cred_hist_length': cred_hist, 'credit_score': credit_score,
            'person_home_ownership': home_own, 'loan_intent': loan_intent,
            'previous_loan_defaults_on_file': prev_default
        }])

        pred  = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='glass' style='text-align:center;padding:30px;'>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<div class='approved'>✅ LOAN APPROVED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='denied'>❌ LOAN DENIED</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown(pbar(proba[1], 'green') + pbar(proba[0], 'red'), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sh'>⚡ SHAP — Why this decision?</div>", unsafe_allow_html=True)
        inp_np = preprocessing.transform(input_df)
        exp    = shap.TreeExplainer(rf)
        sv     = exp(inp_np)
        vals   = sv.values[0,:,1] if sv.values.ndim == 3 else sv.values[0]
        vals_flat = np.array(vals).flatten()
        fn_sh = feat_names[:len(vals_flat)] if len(feat_names) > len(vals_flat) else feat_names
        vals_flat = vals_flat[:len(fn_sh)]
        df_sh  = pd.DataFrame({'feature': fn_sh, 'shap': vals_flat.tolist()})
        df_sh  = df_sh.reindex(df_sh['shap'].abs().sort_values(ascending=False).index).head(12)
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in df_sh['shap']]
        st.plotly_chart(
            plotly_hbar(df_sh['shap'].tolist(), df_sh['feature'].tolist(), colors,
                        "🟢 Green = Towards Approval  |  🔴 Red = Towards Denial", 420),
            use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — SHAP
# ══════════════════════════════════════════════════════════════
elif page == "📊 SHAP Analysis":
    st.markdown("<div class='htitle'>📊 SHAP Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hsub'>Global & Local Feature Attribution</div>", unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        exp   = shap.TreeExplainer(rf)
        samp  = x_test_np[:300]
        
        sv = exp.shap_values(samp)

        # Handle all SHAP formats safely
        if isinstance(sv, list):
            shap_v = sv[1]
        else:
            shap_v = sv

        # Fix shape issues
        if len(shap_v.shape) == 3:
            shap_v = shap_v[:, :, 1]

        shap_v = np.array(shap_v)

    t1, t2, t3 = st.tabs(["📊 Global Bar", "🐝 Beeswarm", "💧 Waterfall"])

    with t1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>Global Feature Importance</div>", unsafe_allow_html=True)
        mean_s = np.abs(shap_v).mean(axis=0)
        if mean_s.ndim > 1: mean_s = mean_s.mean(axis=-1)
        mean_s = np.array(mean_s).flatten()
        # Fix: match lengths
        fn_g = feat_names[:len(mean_s)] if len(feat_names) > len(mean_s) else feat_names
        ms_g = mean_s[:len(fn_g)]
        df_g   = pd.DataFrame({'feature': fn_g, 'importance': ms_g.tolist()})
        df_g   = df_g.sort_values('importance', ascending=True).tail(14)
        fig = go.Figure(go.Bar(
            x=df_g['importance'], y=df_g['feature'], orientation='h',
            marker=dict(color=df_g['importance'], colorscale='Viridis', showscale=True),
            text=[f"{v:.4f}" for v in df_g['importance']], textposition='outside'
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', height=520, margin=dict(t=10,b=10),
                          xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Mean |SHAP|'),
                          yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>Beeswarm Plot</div>", unsafe_allow_html=True)
        fig_b, ax = plt.subplots(figsize=(12,7))
        fig_b.patch.set_facecolor('none'); ax.set_facecolor('none')
        shap.summary_plot(shap_v, samp, feature_names=feat_names, show=False, max_display=14)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        buf.seek(0); st.image(buf, use_container_width=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>Waterfall — Why was this loan DENIED?</div>", unsafe_allow_html=True)
        ypred = rf.predict(x_test_np)
        didx  = int(np.where(ypred == 0)[0][0])
        svs   = exp.shap_values(x_test_np[didx:didx+1])
        svals = svs[1][0] if isinstance(svs, list) else svs[0]
        prob  = rf.predict_proba(x_test_np[didx:didx+1])[0][1]
        svals = np.array(svals).flatten()
        # Fix: trim feat_names to match svals length
        fn_wf = feat_names[:len(svals)] if len(feat_names) > len(svals) else feat_names
        if len(svals) > len(fn_wf): svals = svals[:len(fn_wf)]
        df_wf = pd.DataFrame({'feature': fn_wf, 'shap': svals.tolist()})
        df_wf = df_wf.reindex(df_wf['shap'].abs().sort_values(ascending=False).index).head(12)
        colors= ['#2ecc71' if v > 0 else '#e74c3c' for v in df_wf['shap']]
        st.plotly_chart(plotly_hbar(df_wf['shap'].tolist(), df_wf['feature'].tolist(), colors, height=440), use_container_width=True)
        st.markdown(f"<div class='ibox'>🔴 Red = Denial &nbsp;|&nbsp; 🟢 Green = Approval &nbsp;|&nbsp; Approval Prob: <b>{prob*100:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — LIME
# ══════════════════════════════════════════════════════════════
elif page == "🟡 LIME Explanation":
    st.markdown("<div class='htitle'>🟡 LIME Explanation</div>", unsafe_allow_html=True)
    st.markdown("<div class='hsub'>Local Interpretable Model-agnostic Explanations</div>", unsafe_allow_html=True)

    with st.spinner("Initializing LIME..."):
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=x_train_np, feature_names=feat_names,
            class_names=['Denied','Approved'], mode='classification',
            discretize_continuous=True, random_state=42)

    ypred = rf.predict(x_test_np)
    d_idx = int(np.where(ypred == 0)[0][0])
    a_idx = int(np.where(ypred == 1)[0][0])

    c1, c2 = st.columns(2)
    for col, idx, label in [(c1, d_idx, "❌ DENIED"), (c2, a_idx, "✅ APPROVED")]:
        with col:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>{label} Explanation</div>", unsafe_allow_html=True)
            with st.spinner("Computing..."):
                exp_l = lime_exp.explain_instance(
                    x_test_np[idx], rf.predict_proba,
                    num_features=10, num_samples=500)
            proba = exp_l.predict_proba
            st.markdown(f"<div class='ibox'>Denied: <b style='color:#e74c3c'>{proba[0]*100:.1f}%</b> &nbsp;|&nbsp; Approved: <b style='color:#2ecc71'>{proba[1]*100:.1f}%</b></div>", unsafe_allow_html=True)
            data   = exp_l.as_list()
            colors = ['#2ecc71' if w > 0 else '#e74c3c' for _, w in data]
            fig    = go.Figure(go.Bar(
                x=[w for _,w in data[::-1]], y=[f for f,_ in data[::-1]],
                orientation='h', marker_color=colors[::-1],
                text=[f"{w:+.4f}" for _,w in data[::-1]], textposition='outside'
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white', height=400, margin=dict(t=10,b=10),
                              xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                              yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — DiCE
# ══════════════════════════════════════════════════════════════
elif page == "🔵 DiCE Counterfactuals":
    st.markdown("<div class='htitle'>🔵 DiCE Counterfactuals</div>", unsafe_allow_html=True)
    st.markdown("<div class='hsub'>Minimum changes to flip DENIED → APPROVED</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass'>
        <div class='sh'>❓ What are Counterfactuals?</div>
        <p style='color:#ccc;line-height:1.8;'>DiCE generates <b>alternative scenarios</b> — actionable advice on what to change to get loan approved!</p>
        <div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;'>
            <div class='ibox' style='flex:1;min-width:140px;'>🔷 <b>SHAP</b> — "Why denied?"</div>
            <div class='ibox' style='flex:1;min-width:140px;'>🟡 <b>LIME</b> — "Which features?"</div>
            <div class='ibox' style='flex:1;min-width:140px;'>🔵 <b>DiCE</b> — "What to change?"</div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        with st.spinner("Setting up DiCE..."):
            train_df = x_train.copy()
            train_df['loan_status'] = y_train.values
            cont_feats = ['person_age','person_income','person_emp_exp','loan_amnt',
                          'loan_int_rate','loan_percent_income',
                          'cb_person_cred_hist_length','credit_score']
            cat_feats  = ['person_home_ownership','loan_intent','previous_loan_defaults_on_file']

            d_dice = dice_ml.Data(dataframe=train_df,
                                  continuous_features=cont_feats,
                                  outcome_name='loan_status')
            m_dice = dice_ml.Model(model=pipe, backend='sklearn')
            dice   = Dice(d_dice, m_dice, method='random')

            ypred_raw = pipe.predict(x_test)
            denied_raw= x_test[ypred_raw == 0].reset_index(drop=True)
            query     = denied_raw.iloc[[0]]
            pred_prob = pipe.predict_proba(query)[0][1]

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>👤 Current Profile — DENIED Applicant</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            for col in cont_feats[:4]:
                if col in query.columns:
                    st.metric(col.replace('_',' ').title(), f"{query[col].values[0]:.1f}")
        with c2:
            for col in cont_feats[4:]:
                if col in query.columns:
                    st.metric(col.replace('_',' ').title(), f"{query[col].values[0]:.1f}")
        st.markdown(f"<div class='denied' style='margin-top:12px;'>❌ DENIED — Approval Prob: {pred_prob*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        num_cfs = st.slider("Number of Counterfactuals", 1, 5, 3)

        if st.button("🔵 Generate Counterfactuals"):
            with st.spinner("Generating..."):
                cf = dice.generate_counterfactuals(
                    query, total_CFs=num_cfs, desired_class="opposite",
                    permitted_range={
                        'credit_score': [300,850], 'person_income': [10000,500000],
                        'loan_int_rate': [1.0,25.0], 'loan_percent_income': [0.01,0.90],
                        'loan_amnt': [500,35000]
                    },
                    features_to_vary=['credit_score','person_income','loan_amnt',
                                      'loan_int_rate','loan_percent_income',
                                      'person_home_ownership','loan_intent',
                                      'previous_loan_defaults_on_file']
                )
                cf_df = cf.cf_examples_list[0].final_cfs_df.reset_index(drop=True)

            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='sh'>✅ Actionable Changes</div>", unsafe_allow_html=True)
            for idx, row in cf_df.iterrows():
                st.markdown(f"<b style='color:#4ecdc4;'>🔄 Scenario #{idx+1} — APPROVED ✅</b>", unsafe_allow_html=True)
                for col in cont_feats:
                    if col in query.columns and col in cf_df.columns:
                        orig, new = query[col].values[0], row[col]
                        if abs(orig - new) > 0.5:
                            arrow = "⬆️" if new > orig else "⬇️"
                            st.markdown(f"&nbsp;&nbsp;&nbsp;{arrow} **{col.replace('_',' ').title()}**: `{orig:.1f}` → `{new:.1f}`")
                for col in cat_feats:
                    if col in query.columns and col in cf_df.columns:
                        if str(query[col].values[0]) != str(row[col]):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;🔀 **{col.replace('_',' ').title()}**: `{query[col].values[0]}` → `{row[col]}`")
                st.markdown("---")

            changed = [c for c in cont_feats if c in cf_df.columns and c in query.columns
                       and (cf_df[c] - query[c].values[0]).abs().max() > 0.5]
            if changed:
                st.markdown("<div class='sh'>📊 Visual Comparison</div>", unsafe_allow_html=True)
                chart_cols = st.columns(min(len(changed[:4]), 4))
                for cc, feat in zip(chart_cols, changed[:4]):
                    with cc:
                        ov = query[feat].values[0]
                        cv = cf_df[feat].values
                        fig = go.Figure(go.Bar(
                            x=['Original\n(DENIED)'] + [f'CF{i+1}\n(APPROVED)' for i in range(len(cv))],
                            y=[ov]+list(cv),
                            marker_color=['#e74c3c']+['#2ecc71']*len(cv),
                            text=[f"{v:.1f}" for v in [ov]+list(cv)],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title=feat.replace('_',' ').title(),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', height=280, margin=dict(t=40,b=10),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"DiCE Error: {e}")
        st.info("Run: pip install dice-ml")
