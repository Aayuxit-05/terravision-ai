import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TerraVision AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("terravision_model.keras")


model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.12), transparent 28%),
        radial-gradient(circle at 85% 25%, rgba(34, 197, 94, 0.10), transparent 25%),
        radial-gradient(circle at 50% 80%, rgba(14, 116, 144, 0.15), transparent 35%),
        linear-gradient(180deg, #030817 0%, #061526 48%, #031017 100%);
    color: #f1f5f9;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Animated stars */
.stApp::before {
    content: "✦   ·       ✧        ·   ✦        ·       ✧    ·      ✦       ·        ✧";
    position: fixed;
    top: 30px;
    left: 0;
    width: 100%;
    color: rgba(255,255,255,0.45);
    font-size: 16px;
    letter-spacing: 35px;
    line-height: 80px;
    pointer-events: none;
    z-index: 0;
    animation: stars 8s ease-in-out infinite alternate;
}

@keyframes stars {
    from {
        opacity: 0.35;
        transform: translateY(0px);
    }
    to {
        opacity: 0.8;
        transform: translateY(12px);
    }
}

/* Hero */
.hero {
    text-align: center;
    padding: 35px 20px 20px;
    position: relative;
}

.hero-orbit {
    font-size: 46px;
    display: inline-block;
    animation: float 4s ease-in-out infinite;
    filter: drop-shadow(0 0 18px rgba(56,189,248,0.5));
}

@keyframes float {
    0%, 100% {
        transform: translateY(0px) rotate(-4deg);
    }
    50% {
        transform: translateY(-10px) rotate(4deg);
    }
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    margin: 8px 0 4px;
    background: linear-gradient(90deg, #f8fafc, #7dd3fc, #86efac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a7c7d8;
    font-size: 1.15rem;
    margin-bottom: 8px;
}

/* Earth horizon */
.earth-horizon {
    height: 100px;
    margin: 25px -20px 35px;
    overflow: hidden;
    position: relative;
}

.earth {
    width: 120%;
    height: 190px;
    margin-left: -10%;
    border-radius: 50% 50% 0 0;
    background:
        radial-gradient(circle at 25% 35%, rgba(74,222,128,0.85) 0 5%, transparent 6%),
        radial-gradient(circle at 55% 25%, rgba(34,197,94,0.7) 0 7%, transparent 8%),
        radial-gradient(circle at 72% 50%, rgba(16,185,129,0.75) 0 6%, transparent 7%),
        linear-gradient(180deg, #075985, #0369a1 45%, #064e3b);
    box-shadow:
        0 -10px 40px rgba(56,189,248,0.45),
        inset 0 15px 35px rgba(255,255,255,0.08);
    animation: earthmove 12s ease-in-out infinite alternate;
}

@keyframes earthmove {
    from {
        transform: translateY(8px);
    }
    to {
        transform: translateY(-4px);
    }
}

/* Upload card */
.upload-card {
    background: rgba(10, 31, 43, 0.72);
    border: 1px solid rgba(125,211,252,0.20);
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 20px 70px rgba(0,0,0,0.25);
    backdrop-filter: blur(14px);
    margin-bottom: 25px;
}

.upload-title {
    text-align: center;
    font-size: 1.45rem;
    font-weight: 600;
    color: #e0f2fe;
}

.upload-subtitle {
    text-align: center;
    color: #91aebb;
    margin-bottom: 22px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(4, 20, 30, 0.65);
    border: 1px dashed rgba(125,211,252,0.4);
    border-radius: 18px;
    padding: 10px;
}

/* Result cards */
.result-card {
    background: linear-gradient(
        145deg,
        rgba(15, 54, 58, 0.78),
        rgba(7, 30, 40, 0.82)
    );
    border: 1px solid rgba(134,239,172,0.20);
    border-radius: 22px;
    padding: 28px;
    min-height: 280px;
}

.result-label {
    color: #8faab7;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    color: #bbf7d0;
    margin: 8px 0 18px;
}

.confidence {
    font-size: 2rem;
    font-weight: 700;
    color: #7dd3fc;
}

/* Nature section */
.nature {
    text-align: center;
    margin: 45px 0 10px;
    color: #7897a4;
    font-size: 26px;
    letter-spacing: 14px;
    animation: naturefloat 5s ease-in-out infinite alternate;
}

@keyframes naturefloat {
    from {
        transform: translateY(0);
    }
    to {
        transform: translateY(-5px);
    }
}

/* Info cards */
.info-card {
    background: rgba(8, 29, 39, 0.65);
    border: 1px solid rgba(125,211,252,0.12);
    border-radius: 18px;
    padding: 22px;
    height: 100%;
}

.info-card h3 {
    color: #bae6fd;
}

.info-card p {
    color: #91aebb;
    line-height: 1.65;
}

/* Disclaimer */
.disclaimer {
    margin-top: 25px;
    padding: 14px 18px;
    border-radius: 14px;
    background: rgba(30,41,59,0.55);
    border: 1px solid rgba(148,163,184,0.12);
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 45px;
    color: #587481;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-orbit">🛰️</div>

<h1>TerraVision AI</h1>

<p>See Earth through the eyes of deep learning.</p>

</div>
""", unsafe_allow_html=True)

# Earth horizon
st.markdown("""
<div class="earth-horizon">
    <div class="earth"></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# UPLOAD AREA
# =========================================================

st.markdown("""
<div class="upload-card">

<div class="upload-title">
🌍 Explore a piece of Earth
</div>

<div class="upload-subtitle">
Upload a satellite image and let TerraVision identify its land-cover type.
</div>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">Satellite image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with col2:

        resized_image = image.resize((128, 128))

        image_array = np.array(resized_image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        predicted_class = np.argmax(prediction[0])

        confidence = (
            prediction[0][predicted_class] * 100
        )

        predicted_name = class_names[predicted_class]

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">TerraVision analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-name">🌿 {predicted_name}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">{confidence:.1f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="color:#91aebb;margin-bottom:20px;">'
            'model confidence'
            '</div>',
            unsafe_allow_html=True
        )

        if confidence >= 80:
            st.success(
                "🟢 High confidence — the model strongly favors this category."
            )

        elif confidence >= 50:
            st.warning(
                "🟡 Moderate confidence — the model sees some uncertainty."
            )

        else:
            st.error(
                "🔴 Low confidence — treat this prediction cautiously."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

# =========================================================
# NATURE DECORATION
# =========================================================

st.markdown("""
<div class="nature">
🏔️  🌲  🌲  🏔️  ~  🌊  ~  🌲  🏔️
</div>
""", unsafe_allow_html=True)

# =========================================================
# INFORMATION
# =========================================================

st.markdown("### 🌎 What can TerraVision recognize?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
    <h3>🌲 Natural landscapes</h3>
    <p>
    Forests, vegetation, pasture and different types
    of agricultural land.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
    <h3>🌊 Water & coast</h3>
    <p>
    Rivers and large water bodies such as lakes and
    coastal areas.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
    <h3>🏙️ Human landscapes</h3>
    <p>
    Residential areas, highways and industrial
    regions.
    </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================

st.markdown("### 🧠 How TerraVision works")

st.markdown("""
<div class="info-card">

<p>
TerraVision AI uses <b>deep learning and transfer learning</b>
to recognize land-use and land-cover patterns in satellite imagery.
</p>

<p>
The model was trained on the <b>EuroSAT RGB dataset</b>,
containing 27,000 labeled satellite image patches across
10 categories.
</p>

<p>
This prototype achieved approximately <b>92% validation accuracy</b>
on its held-out validation data.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">
⚠️ <b>Important:</b> TerraVision is an educational AI prototype.
Its predictions are based on patterns learned from the EuroSAT
dataset and should not be treated as professional satellite
imagery analysis. Uploaded images may differ from the training data.
</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
🌌 TerraVision AI &nbsp; • &nbsp; Deep Learning for Earth Observation
<br>
Built as a first-year B.Tech AI/ML project
</div>
""", unsafe_allow_html=True)