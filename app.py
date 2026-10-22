import os
import joblib
import pandas as pd
import streamlit as st
from feature_extractor import extract_features, features_to_df
from train_model import MODEL_SAVE_PATH, METADATA_SAVE_PATH, train_and_evaluate

# Page configuration 
st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 28px 32px; border-radius: 14px; color: white;
    margin-bottom: 28px; border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px; }
.header p  { font-size: 1rem; color: #94a3b8; margin: 0; }

.verdict-phishing {
    background: rgba(239,68,68,0.08); border: 2px solid #ef4444;
    border-radius: 12px; padding: 22px 28px; text-align: center; margin: 18px 0;
}
.verdict-safe {
    background: rgba(16,185,129,0.08); border: 2px solid #10b981;
    border-radius: 12px; padding: 22px 28px; text-align: center; margin: 18px 0;
}
.verdict-phishing h2 { color: #ef4444; margin: 0 0 6px; font-size: 1.5rem; }
.verdict-safe    h2 { color: #10b981; margin: 0 0 6px; font-size: 1.5rem; }
.verdict-phishing p, .verdict-safe p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(METADATA_SAVE_PATH):
        train_and_evaluate()
    return joblib.load(MODEL_SAVE_PATH), joblib.load(METADATA_SAVE_PATH)

model, metadata = load_model()


# Header
st.markdown("""
<div class="header">
  <h1>🛡️ Phishing URL Detector</h1>
  <p>Enter any URL below. Our ML model will analyse it and tell you if it's safe or suspicious.</p>
</div>
""", unsafe_allow_html=True)


# URL Input
url_input  = st.text_input("Enter URL:", placeholder="https://example.com", label_visibility="collapsed")
analyze_btn = st.button("🔍 Analyse URL", type="primary", use_container_width=True)


# ── Analysis & Result ─────────────────────────────────────────────────────────
if analyze_btn:
    raw = url_input.strip()

    if not raw:
        st.warning("Please enter a URL before clicking Analyse.")
    else:
        with st.spinner("Extracting features and running ML inference…"):
            # Step 1: Extract 20 features from the URL
            features, meta = extract_features(raw)

            # Step 2: Convert to DataFrame and predict
            df_feat    = features_to_df(features)
            prediction = model.predict(df_feat)[0]          # 0 = safe, 1 = phishing
            probs      = model.predict_proba(df_feat)[0]    # [prob_safe, prob_phishing]
            phish_pct  = round(probs[1] * 100, 1)

        # ── Verdict card ──────────────────────────────────────────────────────
        if prediction == 1:
            st.markdown("""
            <div class="verdict-phishing">
                <h2>⚠️ PHISHING URL DETECTED</h2>
                <p>This link shows strong signs of a malicious or fraudulent website.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-safe">
                <h2>✅ SAFE / LEGITIMATE URL</h2>
                <p>This link matches patterns of a legitimate, trustworthy website.</p>
            </div>""", unsafe_allow_html=True)

        # ── Risk score bar ────────────────────────────────────────────────────
        st.markdown(f"**Risk Score:** `{phish_pct}%` phishing probability")
        st.progress(float(probs[1]))
        st.markdown("---")

        # ── Risk factors and safe signals ─────────────────────────────────────
        red_flags, green_flags = [], []

        if features['num_suspicious_keywords'] > 0:
            red_flags.append(f"Contains **{features['num_suspicious_keywords']} suspicious keyword(s)**: `{meta['found_keywords']}`")
        if features['has_brand_in_subdomain_or_path']:
            brand = f"'{meta['matched_brand']}'" if meta['matched_brand'] else "a known brand"
            red_flags.append(f"**Brand impersonation** – {brand} found outside its official domain")
        if features['has_ip_address']:
            red_flags.append(f"**Raw IP address** used as host: `{meta['hostname']}`")
        if features['has_at_symbol']:
            red_flags.append("**`@` symbol** in URL — commonly used to disguise the real host")
        if features['has_shortener']:
            red_flags.append("**URL shortener** detected — hides the final destination")
        if not features['has_https']:
            red_flags.append("**Unencrypted HTTP** — no SSL/TLS protection")
        if features['url_entropy'] > 4.4:
            red_flags.append(f"**High URL entropy** ({features['url_entropy']}) — possible obfuscation")
        if features['digit_ratio'] > 0.15:
            red_flags.append(f"**High digit ratio** ({features['digit_ratio']*100:.1f}%) — abnormal numeric density")

        if features['has_https']:
            green_flags.append("Secure **HTTPS** connection")
        if features['num_suspicious_keywords'] == 0:
            green_flags.append("No phishing trigger keywords found")
        if not features['has_brand_in_subdomain_or_path']:
            green_flags.append("No brand spoofing detected")
        if not features['has_ip_address']:
            green_flags.append("Uses a proper domain name, not a raw IP")

        col_r, col_g = st.columns(2)
        with col_r:
            st.markdown("#### 🚨 Risk Indicators")
            if red_flags:
                for flag in red_flags:
                    st.error(flag, icon="🚨")
            else:
                st.success("No risk indicators found.", icon="🛡️")
        with col_g:
            st.markdown("#### ✅ Safe Signals")
            if green_flags:
                for flag in green_flags:
                    st.info(flag, icon="✔️")
            else:
                st.warning("Few standard safe characteristics detected.", icon="⚠️")

        # ── Expandable raw feature table ──────────────────────────────────────
        with st.expander("🔬 View all 20 extracted features"):
            feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)


# Footer 
st.markdown(
    "<br><div style='text-align:center;color:#94a3b8;font-size:0.82rem;'>"
    "🛡️ Phishing URL Detector · Built with Python & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
