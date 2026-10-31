import re
import joblib
import pandas as pd
import streamlit as st

from urllib.parse import urlparse
from url_features import (
    extract_url_features,
    detect_lookalike_and_threats,
    validate_url_input,
    check_domain_dns,
)


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="PhishGuard AI | Website Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==============================================================================
# LOAD TRAINED MODEL
# ==============================================================================

@st.cache_resource
def load_model():
    """Load the pre-trained Random Forest model and feature list."""
    model = joblib.load("phishing_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names


model, feature_names = load_model()


# ==============================================================================
# CUSTOM CSS (DARK NAVY & TURQUOISE THEME)
# ==============================================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #07111f;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    display: none;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    max-width: 1100px;
    padding: 2rem 1.5rem 2rem;
}


/* ==========================================
   HERO SECTION
   ========================================== */

.hero {
    background:
        linear-gradient(
            90deg,
            rgba(5, 13, 27, 0.96) 0%,
            rgba(5, 13, 27, 0.84) 55%,
            rgba(5, 13, 27, 0.48) 100%
        ),
        url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1800&q=85");

    background-size: cover;
    background-position: center;
    min-height: 380px;
    border: 1px solid #20344b;
    border-radius: 24px;
    padding: 45px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 15px 55px rgba(0, 0, 0, 0.25);
    box-sizing: border-box;
}

.brand {
    color: #5eead4;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 28px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(34px, 5vw, 52px);
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 18px 0;
}

.highlight {
    color: #5eead4;
}

.hero p {
    color: #d1dbea;
    font-size: 16px;
    line-height: 1.8;
    max-width: 590px;
    margin: 0;
}


/* ==========================================
   SCANNER CARD
   ========================================== */

.scanner-card {
    background: linear-gradient(
        145deg,
        #111f32,
        #0b1728
    );

    border: 1px solid #263b52;
    border-radius: 22px;
    padding: 28px 30px 30px;
    margin-top: 25px;
    margin-bottom: 0;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
    box-sizing: border-box;
}

.scanner-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin: 0 0 8px 0;
}

.scanner-description {
    color: #9fb0c6;
    font-size: 14.5px;
    line-height: 1.7;
    margin: 0;
}


/* ==========================================
   URL INPUT
   ========================================== */

div[data-testid="stTextInput"] {
    margin-top: 0;
}

div[data-testid="stTextInput"] label {
    display: none;
}

div[data-baseweb="input"] {
    background: #081321;
    border: 1px solid #344a62;
    border-radius: 12px;
    min-height: 54px;
}

div[data-baseweb="input"]:focus-within {
    border-color: #5eead4;
    box-shadow: 0 0 0 1px #5eead4;
}

div[data-baseweb="input"] input {
    color: #ffffff;
    background: transparent;
    font-size: 15px;
}

div[data-baseweb="input"] input::placeholder {
    color: #7f93ad;
}


/* ==========================================
   SCAN BUTTON
   ========================================== */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(
        90deg,
        #2dd4bf,
        #06b6d4
    );

    color: #061521;
    border: none;
    border-radius: 12px;
    min-height: 53px;
    font-size: 16px;
    font-weight: 800;
    transition: all 0.2s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(
        90deg,
        #5eead4,
        #67e8f9
    );

    color: #061521;
    border: none;
    box-shadow: 0 0 24px rgba(45, 212, 191, 0.22);
    transform: translateY(-1px);
}


/* ==========================================
   FORM STYLING
   ========================================== */

div[data-testid="stForm"] {
    border: none;
    background: transparent;
    padding: 0;
}


/* ==========================================
   RESULT CARDS (CLEAR & ACCESSIBLE)
   ========================================== */

.result-card {
    border-radius: 20px;
    padding: 24px 26px;
    margin-top: 25px;
    margin-bottom: 20px;
    box-sizing: border-box;
    transition: all 0.2s ease;
}

.result-card.safe {
    background: linear-gradient(145deg, #09202a, #07151f);
    border: 1px solid #14b8a6;
    box-shadow: 0 12px 35px rgba(20, 184, 166, 0.12);
}

.result-card.suspicious {
    background: linear-gradient(145deg, #231118, #170b10);
    border: 1px solid #f43f5e;
    box-shadow: 0 12px 35px rgba(244, 63, 94, 0.12);
}

.result-card.unverified {
    background: linear-gradient(145deg, #241c0e, #171208);
    border: 1px solid #f59e0b;
    box-shadow: 0 12px 35px rgba(245, 158, 11, 0.12);
}

.result-badge {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.result-badge.safe {
    color: #2dd4bf;
}

.result-badge.suspicious {
    color: #fb7185;
}

.result-badge.unverified {
    color: #fbbf24;
}

.result-summary {
    color: #f1f5f9;
    font-size: 16px;
    font-weight: 600;
    line-height: 1.6;
    margin-bottom: 10px;
}

.result-explanation {
    color: #cbd5e1;
    font-size: 14.5px;
    line-height: 1.6;
    margin-bottom: 12px;
}

.result-reasons-box {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.6;
    border-left: 3px solid #f43f5e;
}

.result-reasons-box ul {
    margin: 6px 0 0 18px;
    padding: 0;
}

.result-warning-box {
    background: rgba(0, 0, 0, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 12px;
    color: #e2e8f0;
    font-size: 13.5px;
    line-height: 1.6;
}

.result-advice {
    color: #94a3b8;
    font-size: 13.5px;
    line-height: 1.6;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 12px;
}


/* ==========================================
   SAFETY NOTE AND FOOTER
   ========================================== */

.safety-note {
    background: rgba(15, 29, 47, 0.8);
    border: 1px solid #23374d;
    border-radius: 12px;
    padding: 15px 18px;
    margin-top: 22px;
    color: #9fb0c6;
    font-size: 13px;
    line-height: 1.7;
}

.footer {
    color: #71839a;
    text-align: center;
    font-size: 12px;
    padding-top: 25px;
    padding-bottom: 10px;
}


/* ==========================================
   MOBILE LAYOUT
   ========================================== */

@media (max-width: 640px) {

    .block-container {
        padding: 1rem 0.8rem;
    }

    .hero {
        padding: 30px 23px;
        min-height: 340px;
    }

    .scanner-card {
        padding: 23px 18px;
    }

    .hero p {
        font-size: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HERO SECTION
# ==============================================================================

hero_html = """
<div class="hero">

    <div class="brand">🛡️ PHISHGUARD AI</div>

    <h1>
        Browse Smarter.<br>
        Stay <span class="highlight">Safer.</span>
    </h1>

    <p>
        Check any website link to see if it's safe to visit.
        Enter a web address below to protect yourself from scams and fake websites.
    </p>

</div>
"""

st.html(hero_html)


# ==============================================================================
# SCANNER HEADING
# ==============================================================================

scanner_html = """
<div class="scanner-card">

    <div class="scanner-title">
        🔍 Website Security Scanner
    </div>

    <div class="scanner-description">
        Type or paste the website address you'd like to check.
        For example: google.com or https://example.com
    </div>

</div>
"""

st.html(scanner_html)


# ==============================================================================
# URL INPUT FORM
# ==============================================================================

with st.form("website_scanner", border=False):

    url = st.text_input(
        "Website URL",
        placeholder="https://www.example.com",
        label_visibility="collapsed"
    )

    scan_clicked = st.form_submit_button(
        "🔍  Scan Website",
        use_container_width=True
    )


# ==============================================================================
# SCAN WORKFLOW & HYBRID DECISION ENGINE
# ==============================================================================

if scan_clicked:

    is_valid, error_kind, error_msg, candidate = validate_url_input(url)

    if not is_valid:
        if error_kind == "empty":
            st.warning(error_msg)
        elif error_kind == "email":
            st.info(f"✉️ {error_msg}")
        else:
            st.error(error_msg)

    else:
        try:
            parsed = urlparse(candidate)
            hostname = (parsed.hostname or "").lower()
            scheme_specified = bool(re.match(r"^https?://", url.strip(), re.IGNORECASE))

            st.markdown("### 🔎 Scan Results")

            # ------------------------------------------------------------------
            # STEP 1: LOOKALIKE & SUSPICIOUS DOMAIN ANALYSIS
            # ------------------------------------------------------------------
            is_lookalike_suspicious, lookalike_reasons, lookalike_warnings = detect_lookalike_and_threats(
                hostname, parsed_path=parsed.path
            )

            if is_lookalike_suspicious:
                # Category 2: Suspicious Website (Flagged by Security Rules)
                reasons_list_html = "".join(f"<li>{reason}</li>" for reason in lookalike_reasons)
                suspicious_card_html = f"""
                <div class="result-card suspicious">
                    <div class="result-badge suspicious">🚨 Suspicious Website</div>
                    <div class="result-summary">
                        Be careful — this website looks suspicious.
                    </div>
                    <div class="result-reasons-box">
                        <b>What we noticed:</b>
                        <ul>
                            {reasons_list_html}
                        </ul>
                    </div>
                    <div class="result-advice">
                        🛡️ <b>Safety tip:</b> Avoid entering your password, payment details, or personal information on this page.
                    </div>
                </div>
                """
                st.html(suspicious_card_html)

            else:
                # --------------------------------------------------------------
                # STEP 2: SAFE TIMEOUT-ENABLED ADDRESS VERIFICATION (DNS)
                # --------------------------------------------------------------
                domain_resolves, dns_status = check_domain_dns(hostname, timeout=3.0)

                if not domain_resolves:
                    # Category 3: Website Could Not Be Verified
                    explanation = (
                        "The website address took too long to respond. The site may be experiencing temporary connectivity issues."
                        if dns_status == "timeout"
                        else "The site might be misspelled, temporarily down, or no longer active on the internet."
                    )

                    unverified_card_html = f"""
                    <div class="result-card unverified">
                        <div class="result-badge unverified">⚠️ Website Could Not Be Verified</div>
                        <div class="result-summary">
                            We couldn't reach or verify this website address.
                        </div>
                        <div class="result-explanation">
                            {explanation}
                        </div>
                        <div class="result-advice">
                            💡 <b>Tip:</b> Please check the spelling of the address and try again.
                        </div>
                    </div>
                    """
                    st.html(unverified_card_html)

                else:
                    # ----------------------------------------------------------
                    # STEP 3: MACHINE LEARNING MODEL PREDICTION
                    # ----------------------------------------------------------
                    features = extract_url_features(
                        candidate,
                        dns_resolved=True,
                        scheme_specified=scheme_specified
                    )

                    # Pass features matching the exact training dataset schema
                    input_data = pd.DataFrame(
                        [[features[name] for name in feature_names]],
                        columns=feature_names
                    )

                    prediction = int(model.predict(input_data)[0])  # 1 = Phishing, 0 = Legitimate

                    # ----------------------------------------------------------
                    # STEP 4: HYBRID DECISION
                    # ----------------------------------------------------------
                    if prediction == 1:
                        # Category 2: Suspicious Website (Flagged by ML Classifier)
                        suspicious_ml_card_html = """
                        <div class="result-card suspicious">
                            <div class="result-badge suspicious">🚨 Suspicious Website</div>
                            <div class="result-summary">
                                Be careful — this website looks suspicious.
                            </div>
                            <div class="result-reasons-box">
                                <b>What we noticed:</b>
                                <ul>
                                    <li>This address shows formatting tricks commonly used by fake or misleading websites.</li>
                                </ul>
                            </div>
                            <div class="result-advice">
                                🛡️ <b>Safety tip:</b> If you arrived here from an unexpected message or email, do not log in or share any sensitive details.
                            </div>
                        </div>
                        """
                        st.html(suspicious_ml_card_html)

                    else:
                        # Category 1: Likely Safe
                        warning_html = ""
                        if lookalike_warnings:
                            warning_items = "".join(f"<li>{w}</li>" for w in lookalike_warnings)
                            warning_html = f"""
                            <div class="result-warning-box">
                                <ul>{warning_items}</ul>
                            </div>
                            """

                        safe_card_html = f"""
                        <div class="result-card safe">
                            <div class="result-badge safe">🟢 Likely Safe</div>
                            <div class="result-summary">
                                This website looks safe to visit.
                            </div>
                            {warning_html}
                            <div class="result-explanation">
                                We didn't detect any deceptive patterns, fake branding, or phishing tricks in this web address.
                            </div>
                            <div class="result-advice">
                                💡 <b>Tip:</b> Make sure you recognize the web address before entering passwords or card details.
                            </div>
                        </div>
                        """
                        st.html(safe_card_html)

        except ValueError:
            st.error("Please enter a valid website address.")

        except Exception as e:
            st.error("Unable to analyze this web address. Please check the link and try again.")
            st.caption(f"Details: {e}")


# ==============================================================================
# SAFETY NOTE AND FOOTER
# ==============================================================================

safety_footer_html = """
<div class="safety-note">
    💡 <b>Quick Tip:</b> Always check the address in your browser bar before sharing passwords or payment details.
</div>

<div class="footer">
    PhishGuard AI · Machine Learning-Based Phishing Website Detection
</div>
"""

st.html(safety_footer_html)
