import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH  = "car_damage_model_final.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["minor", "moderate", "no_damage", "severe"]  # alphabetical — must match training

INSURANCE_POLICY = {
    "no_damage": {
        "label"   : "No Damage",
        "emoji"   : "✅",
        "color"   : "#27ae60",
        "bg"      : "#eafaf1",
        "decision": "Claim Rejected",
        "amount"  : "PKR 0",
        "reason"  : "Vehicle shows no visible damage. Insurance claim cannot be processed.",
        "action"  : "No repair required. Drive safely!"
    },
    "minor": {
        "label"   : "Minor Damage",
        "emoji"   : "🟡",
        "color"   : "#f39c12",
        "bg"      : "#fef9e7",
        "decision": "Partial Claim Approved",
        "amount"  : "PKR 15,000 – 50,000",
        "reason"  : "Minor scratches, small dents, or paint damage detected.",
        "action"  : "Visit nearest authorized repair center within 7 days."
    },
    "moderate": {
        "label"   : "Moderate Damage",
        "emoji"   : "🟠",
        "color"   : "#e67e22",
        "bg"      : "#fef5e7",
        "decision": "Claim Approved",
        "amount"  : "PKR 50,000 – 1,50,000",
        "reason"  : "Significant body damage, broken parts, or panel deformation detected.",
        "action"  : "Schedule inspection with insurance surveyor within 48 hours."
    },
    "severe": {
        "label"   : "Severe Damage",
        "emoji"   : "🔴",
        "color"   : "#e74c3c",
        "bg"      : "#fdedec",
        "decision": "Full Claim Approved",
        "amount"  : "PKR 1,50,000 – Total Loss",
        "reason"  : "Major structural damage or total loss condition detected.",
        "action"  : "Emergency surveyor dispatched. Do NOT drive the vehicle."
    }
}

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_damage_model():
    return load_model(MODEL_PATH)

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
def predict(image: Image.Image, model):
    img   = image.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    preds      = model.predict(arr, verbose=0)[0]
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100
    all_probs  = {CLASS_NAMES[i]: float(preds[i]) * 100 for i in range(len(CLASS_NAMES))}
    return pred_class, confidence, all_probs

# ─────────────────────────────────────────────
#  PREDICTION UI
# ─────────────────────────────────────────────
def bnpl_prediction_ui():
    st.subheader("")

    # Load model
    with st.spinner("🔄 Loading AI model..."):
        try:
            model = load_damage_model()
        except Exception as e:
            st.stop()

    # Upload section
    st.markdown("""
    <div class="card">
        <h3 style="margin:0; color:#333;">📤 Upload Vehicle Photos</h3>
        <p style="margin:4px 0 0; color:#666; font-size:0.9rem;">
            Upload 1–5 images (JPG, JPEG, PNG). Each image is analyzed separately.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload vehicle images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("⚠️ Maximum 5 images allowed. Only first 5 will be processed.")
            uploaded_files = uploaded_files[:5]

        st.markdown("---")
        st.markdown(f"### 🔍 Analyzing {len(uploaded_files)} Image(s)...")

        results = []

        for idx, uploaded_file in enumerate(uploaded_files):
            image      = Image.open(uploaded_file)
            pred_class, confidence, all_probs = predict(image, model)
            policy     = INSURANCE_POLICY[pred_class]

            results.append({
                "file"      : uploaded_file.name,
                "class"     : pred_class,
                "label"     : policy["label"],
                "emoji"     : policy["emoji"],
                "confidence": confidence,
                "decision"  : policy["decision"],
                "amount"    : policy["amount"],
                "color"     : policy["color"]
            })

            st.markdown(f"""
            <div class="card">
                <h4 style="margin:0; color:#333;">
                    Image {idx+1} &nbsp;·&nbsp;
                    <span style="font-weight:normal; font-size:0.88rem; color:#888;">
                        {uploaded_file.name}
                    </span>
                </h4>
            </div>
            """, unsafe_allow_html=True)

            col_img, col_res = st.columns([1, 1.6], gap="large")

            with col_img:
                st.image(image, use_container_width=True)

            with col_res:

                # Damage badge
                st.markdown(f"""
                <div style="background:{policy['bg']}; border-left:5px solid {policy['color']};
                            border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
                    <h3 style="margin:0; color:{policy['color']};">
                        {policy['emoji']} {policy['label']}
                    </h3>
                    <p style="margin:4px 0 0; color:#555;">
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown("**📊 Class Probabilities**")
                for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                    lbl = INSURANCE_POLICY[cls]["label"]
                    st.markdown(
                        f'<p style="margin:0; font-size:0.82rem; color:#555;">'
                        f'{lbl}: <strong>{prob:.1f}%</strong></p>',
                        unsafe_allow_html=True
                    )
                    st.progress(prob / 100)

                # Insurance decision box
                st.markdown(f"""
                <div class="card results" style="border-top:3px solid {policy['color']}; margin-top:0.8rem;">
                    <h4 style="margin:0 0 0.6rem; color:{policy['color']};">
                        🏦 Insurance Decision
                    </h4>
                    <p style="margin:0.2rem 0;"><strong>Decision:</strong> {policy['decision']}</p>
                    <p style="margin:0.2rem 0;">
                        <strong>Coverage Estimate:</strong>&nbsp;
                        <span style="background:#1a1a2e; color:#fff; padding:3px 14px;
                                     border-radius:50px; font-size:0.92rem;">
                            {policy['amount']}
                        </span>
                    </p>
                    <p style="margin:0.6rem 0 0.2rem; color:#555; font-size:0.88rem;">
                        📌 {policy['reason']}
                    </p>
                    <p style="background:{policy['bg']}; border-radius:8px;
                              padding:0.5rem 0.9rem; font-size:0.88rem; margin:0.4rem 0 0;">
                        <strong>Next Step:</strong> {policy['action']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 2rem; background:#ffffff;
                    border-radius:12px; border:2px dashed #ddd; margin-top:1rem;">
            <h2>📸</h2>
            <h3 style="color:#aaa;">No images uploaded yet</h3>
            <p style="color:#ccc;">Upload 1–5 vehicle photos above to get your instant damage report</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  YOUR ORIGINAL PAGE WRAPPER
# ─────────────────────────────────────────────
def theory_subject_page():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #f7f9fc, #eef2f7);
        color: #333333;
    }
    .stButton>button {
        background-color: #f1f1f1;
        color: #333333;
        border-radius: 12px;
        padding: 10px;
        font-size: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e0e0e0;
        transform: scale(1.05);
    }
    .stMarkdown>p {
        text-align: justify;
        color: #333;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stHeader>h1 {
        color: #eab676;
        font-weight: bold;
        font-size: 30px;
        text-align: center;
    }
    .stSubheader>h2 {
        font-size: 22px;
        color: #388e3c;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .results {
        transition: background-color 0.3s ease;
    }
    .results:hover {
        background-color: #fff3e0;
    }
    .stMarkdown ul { list-style-type: none; padding-left: 0; }
    .stMarkdown li { padding: 5px 0; }
    .stTextInput>label { font-size: 16px; color: #eab676; }
    .stIcon { color: black !important; }
    </style>
    """, unsafe_allow_html=True)

    bnpl_prediction_ui()

    # Sidebar
    try:
        st.sidebar.image("logo.png", use_container_width=True)
    except Exception:
        st.sidebar.markdown("## 🚗 AutoShield")

    st.sidebar.title("About Me:")
    st.sidebar.markdown("""
- **Name:** Muhammad Faizan Akram
- **Reg No:** FA23-BBD-090 (6B)
- **Email:** mfakram28@gmail.com
    """)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="VDDI App",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <h1 style="text-align:center; color:#333333; font-weight:bold; font-size:33px;">
        🚗 Vehicle Damage Detection for Insurance
    </h1>
    """, unsafe_allow_html=True)

    theory_subject_page()


if __name__ == "__main__":
    main()