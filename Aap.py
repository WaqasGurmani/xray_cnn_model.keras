

# ===============================
# 1. Imports
# ===============================
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ===============================
# 2. Page Configuration
# ===============================
st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🩻",
    layout="wide"
)

# ===============================
# 3. Custom CSS (Professional UI)
# ===============================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }

.header {
    background: linear-gradient(90deg, #0b5394, #073763);
    padding: 30px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

section[data-testid="stSidebar"] {
    background-color: #e8f0fe;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. Header
# ===============================
st.markdown("""
<div class="header">
<h1>🩻 Chest X-ray Disease Detection</h1>
<p>AI-powered screening & decision support</p>
<b>Developed by Waqas Gurmani</b>
</div>
""", unsafe_allow_html=True)

# ===============================
# 5. Sidebar – Patient Info
# ===============================
st.sidebar.title("🧑‍⚕️ Patient Information")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Patient Age", 0, 120, 25)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This AI system is for educational & screening purposes only.\n"
    "It does NOT replace professional medical diagnosis."
)

# ===============================
# 6. Load Model
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("xray_cnn_model.keras")

model = load_model()
class_names = ["normal", "pneumonia", "tuberculosis"]
IMG_SIZE = (128, 128)

# ===============================
# 7. Upload Image
# ===============================
st.subheader("📤 Upload Chest X-ray Image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# 8. PDF Generator (Clinical Style)
# ===============================
def generate_pdf(report):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(
        temp_file.name,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "<para align='center'><b><font size='16'>Chest X-ray AI Clinical Report</font></b></para>",
        styles["Normal"]
    ))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Patient Information</b>", styles["Heading2"]))
    story.append(Paragraph(report["patient"], styles["Normal"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>AI Screening Results</b>", styles["Heading2"]))
    story.append(Paragraph(report["results"], styles["Normal"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Treatment / Recommended Next Steps</b>", styles["Heading2"]))
    story.append(Paragraph(report["treatment"], styles["Normal"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(
        "<i>This AI-generated report is intended for educational and screening purposes only "
        "and does not replace professional medical diagnosis or treatment.</i>",
        styles["Italic"]
    ))

    story.append(Paragraph("<br/>", styles["Normal"]))
    story.append(Paragraph(
        "<para align='center'>Developed by <b>Engineer Waqas Gurmani</b></para>",
        styles["Normal"]
    ))

    doc.build(story)
    return temp_file.name

# ===============================
# 9. Main Logic
# ===============================
if uploaded_file and patient_name.strip():

    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_res = st.columns([1.3, 1])

    with col_res:
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        confidence_pct = confidence * 100
        predicted_class = class_names[idx]

        # -----------------------------
        # Treatment Text (Single Source)
        # -----------------------------
        if predicted_class == "normal":
            treatment_text = (
                "No abnormal findings detected. "
                "Maintain a healthy lifestyle, avoid smoking and air pollution. "
                "Seek medical advice if symptoms develop."
            )
        elif predicted_class == "pneumonia":
            treatment_text = (
                "Possible pneumonia detected. "
                "Consult a qualified physician promptly. "
                "Further clinical evaluation and tests may be required."
            )
        else:
            treatment_text = (
                "Possible tuberculosis detected. "
                "Immediate medical attention is required. "
                "Confirm diagnosis through laboratory investigations."
            )

        # -----------------------------
        # Clinical Report (UI)
        # -----------------------------
        report = {
            "patient": f"<b>Name:</b> {patient_name}<br/><b>Age:</b> {patient_age} years",
            "results": (
                f"<b>Predicted Condition:</b> {predicted_class.upper()}<br/>"
                f"<b>Confidence Score:</b> {confidence_pct:.2f}%"
            ),
            "treatment": treatment_text
        }

        st.subheader("📋 AI Clinical Report")

        st.markdown("### 🧑‍⚕️ Patient Information")
        st.markdown(report["patient"], unsafe_allow_html=True)

        st.markdown("### 🧠 AI Screening Results")
        st.markdown(report["results"], unsafe_allow_html=True)
        st.progress(confidence)

        st.markdown("### 🩺 Treatment / Next Steps")
        st.markdown(report["treatment"])

        st.info(
            "This AI-generated report is for educational & screening purposes only "
            "and does not replace professional medical diagnosis."
        )

        # -----------------------------
        # PDF Download (Same Report)
        # -----------------------------
        pdf_path = generate_pdf(report)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Report",
                f,
                file_name=f"{patient_name}_Chest_Xray_Report.pdf",
                mime="application/pdf"
            )

    with col_img:
        st.subheader("🖼 X-ray Preview (Optional)")
        if st.button("👁️ Show Uploaded X-ray"):
            st.image(image, use_container_width=True)
        else:
            st.info("X-ray image hidden for privacy.")

else:
    st.info("⬆️ Please enter patient details and upload a chest X-ray image.")

# ===============================
# 10. Footer
# ===============================
st.markdown("""
<div class="footer">
AI Medical Screening Prototype<br>
<b>Developed by  Waqas Gurmani</b>
</div>
""", unsafe_allow_html=True)
