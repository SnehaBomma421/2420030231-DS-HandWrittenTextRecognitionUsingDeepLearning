import os
import sys
import importlib
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pypdfium2 as pdfium

import preprocessing
import segmentation
import recognition
import trocr_recognition
import pdf_generator

importlib.reload(preprocessing)
importlib.reload(segmentation)
importlib.reload(recognition)
importlib.reload(trocr_recognition)
importlib.reload(pdf_generator)

from preprocessing import read_image, preprocess_image, remove_shadows, suppress_bleed_through_and_binarize
from segmentation import segment_characters, get_line_crops, draw_boxes
from recognition import load_mapping, load_emnist_model, recognize_segments, generate_confidence_review
from pdf_generator import generate_pdf
from trocr_recognition import recognize_line, normalize_line_endings, processor as trocr_processor, model as trocr_model
from pathlib import Path
# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="NoteVision AI // AMG Cockpit Edition",
    page_icon="⚡",
    layout="wide"
)

# ==============================
# MERCEDES-AMG COCKPIT STYLING
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

.stApp {
    background-color: #060608;
    background-image: 
        radial-gradient(ellipse at 50% 20%, rgba(255, 51, 0, 0.18) 0%, rgba(10, 10, 14, 0.98) 65%),
        linear-gradient(180deg, #09090d 0%, #030304 100%);
    color: #f1f5f9;
}

/* AMG Cockpit Stadium Header */
.amg-cockpit-banner {
    position: relative;
    max-width: 980px;
    margin: 1rem auto 2.2rem auto;
    padding: 2.2rem 3rem;
    border-radius: 90px;
    background: radial-gradient(ellipse at center, rgba(28, 18, 22, 0.95) 0%, rgba(10, 10, 14, 0.98) 100%);
    border: 2px solid #ff3300;
    box-shadow: 
        0 0 50px rgba(255, 51, 0, 0.4),
        inset 0 0 35px rgba(255, 51, 0, 0.2);
    text-align: center;
}

.amg-top-telemetry {
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    color: #ff6633;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

.amg-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 5px;
    color: #ffffff;
    text-shadow: 0 0 25px rgba(255, 255, 255, 0.5), 0 0 40px rgba(255, 51, 0, 0.4);
    margin: 0;
    text-transform: uppercase;
}

.amg-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ff8866;
    letter-spacing: 4px;
    margin-top: 0.4rem;
    text-transform: uppercase;
}

.telemetry-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 51, 0, 0.25);
}

.telemetry-item {
    text-align: center;
}

.telemetry-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 1px;
}

.telemetry-label {
    font-size: 0.72rem;
    color: #94a3b8;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
}

/* Section Cockpit Titles */
.cockpit-section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #ffffff;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.cockpit-section-title span {
    color: #ff3300;
    text-shadow: 0 0 10px rgba(255, 51, 0, 0.8);
}

/* Line debug card */
.line-debug-card {
    background: rgba(18, 19, 26, 0.9);
    border: 1px solid rgba(255, 51, 0, 0.35);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}

.line-badge {
    display: inline-block;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #ff6633;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}

.line-prediction-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
    background: rgba(0, 0, 0, 0.5);
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    border-left: 3px solid #ff3300;
    margin-top: 0.4rem;
}

/* Text Area Dark Futuristic Styling */
.stTextArea textarea {
    background-color: #0b0c10 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 51, 0, 0.35) !important;
    border-radius: 16px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    padding: 1.2rem !important;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8) !important;
}

.stTextArea textarea:focus {
    border-color: #ff3300 !important;
    box-shadow: 0 0 20px rgba(255, 51, 0, 0.4), inset 0 0 20px rgba(0, 0, 0, 0.8) !important;
}

/* AMG Neon Action Button */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #ff2a00 0%, #b30000 100%) !important;
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif !important;
    border: 1px solid #ff6644 !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 25px rgba(255, 42, 0, 0.5) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    background: linear-gradient(135deg, #ff4411 0%, #d60000 100%) !important;
    box-shadow: 0 0 40px rgba(255, 51, 0, 0.8) !important;
}

/* File Uploader Frame */
div[data-testid="stFileUploader"] {
    background: rgba(15, 16, 22, 0.8) !important;
    border: 1px dashed rgba(255, 51, 0, 0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    color: #94a3b8 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff3300 !important;
    border-bottom-color: #ff3300 !important;
}

hr {
    border-color: rgba(255, 51, 0, 0.25) !important;
    margin: 2.2rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# MODEL & MAPPING PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "emnist_model.keras"
MAPPING_PATH = BASE_DIR / "data" / "emnist_mapping.txt"


@st.cache_resource
def get_ai():
    """Load EMNIST CNN and class mapping once into memory."""
    model = load_emnist_model(MODEL_PATH)
    mapping = load_mapping(MAPPING_PATH)
    return model, mapping


def extract_pages_from_upload(uploaded_file):
    """
    Extracts a list of (page_label, opencv_bgr_image) from an uploaded image or PDF file.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        pages = []
        try:
            pdf = pdfium.PdfDocument(file_bytes)
            for i, page in enumerate(pdf):
                # Render at scale 2.0 (~150-200 DPI)
                pil_image = page.render(scale=2.0).to_pil()
                bgr_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                pages.append((f"{uploaded_file.name} [Page {i + 1}]", bgr_img))
        except Exception as e:
            st.error(f"❌ Error rendering PDF {uploaded_file.name}: {e}")
        return pages
    else:
        img = read_image(file_bytes)
        if img is not None:
            return [(uploaded_file.name, img)]
        return []


# ==============================
# AMG COCKPIT HERO BANNER
# ==============================
st.markdown("""
<div class="amg-cockpit-banner">
    <div class="amg-top-telemetry">
        <span>⚡ SYSTEM ACTIVE // NEURAL HANDWRITING RECONSTRUCTION</span>
    </div>
    <div class="amg-title">NOTEVISION AI</div>
    <div class="amg-subtitle">ADVANCED HANDWRITING MATRIX ENGINE</div>
    <div class="telemetry-bar">
        <div class="telemetry-item">
            <div class="telemetry-value">OPENCV 5.0</div>
            <div class="telemetry-label">SHADOW & RULE REMOVAL</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-value">TrOCR BASE</div>
            <div class="telemetry-label">FULL-LINE RECOGNITION</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-value">EMNIST CNN</div>
            <div class="telemetry-label">CONFIDENCE REVIEW</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-value">PDF EXPORT</div>
            <div class="telemetry-label">VECTOR GENERATOR</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Check model availability
if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
    st.error("❌ EMNIST model or mapping file missing.")
    st.stop()

try:
    emnist_model, emnist_mapping = get_ai()
except Exception as e:
    st.error("❌ Model load error.")
    st.exception(e)
    st.stop()

# ==============================
# MODULE 01 // INPUT STREAM
# ==============================
st.markdown("""
<div class="cockpit-section-title">
    <span>01 //</span> HANDWRITTEN INPUT STREAM
</div>
""", unsafe_allow_html=True)

col_up, col_opt = st.columns([3, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "UPLOAD NOTE PAGES (JPG / PNG / JPEG / WEBP / PDF)",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        accept_multiple_files=True
    )
with col_opt:
    st.markdown("<br>", unsafe_allow_html=True)
    show_debug = st.checkbox("DISPLAY LINE CROPS & TELEMETRY", value=True)
    run_emnist_review = st.checkbox("EMNIST AI CONFIDENCE REVIEW", value=True)

if uploaded_files:
    all_pages = []
    for f in uploaded_files:
        pages = extract_pages_from_upload(f)
        all_pages.extend(pages)

    if not all_pages:
        st.warning("⚠️ No valid image frames could be parsed from upload.")
        st.stop()

    page_transcripts = []

    for page_idx, (page_title, original_img) in enumerate(all_pages, start=1):
        st.markdown(f"### 📄 Page {page_idx}: `{page_title}`")

        with st.spinner(f"⚡ Processing Page {page_idx} Preprocessing & Text-Line Detection..."):
            # 1. Advanced OpenCV Preprocessing
            clean_binary, clean_color, meta = preprocess_image(original_img, return_color=True)

            # 2. Text-line Detection
            characters, boxes, line_ids, line_regions = segment_characters(clean_binary)
            line_crops = get_line_crops(clean_color, line_regions)

        if not line_regions:
            st.warning(f"⚠️ No handwriting lines detected on {page_title}.")
            continue

        # 3. TrOCR Recognition for each detected text line
        line_results = []
        with st.spinner(f"⚡ Running TrOCR Full-Line Recognition on {len(line_crops)} lines..."):
            prev_context = ""
            for l_idx, (crop, (y1, y2, x1, x2)) in enumerate(line_crops):
                try:
                    pred_text = recognize_line(crop, context=prev_context)
                except TypeError:
                    try:
                        pred_text = recognize_line(crop, prev_context)
                    except TypeError:
                        pred_text = recognize_line(crop)
                if pred_text.strip():
                    prev_context = pred_text
                line_results.append({
                    "line_id": l_idx,
                    "crop": crop,
                    "region": (y1, y2, x1, x2),
                    "text": pred_text
                })

        # Filter line results to only those that have valid recognized text
        valid_line_results = [r for r in line_results if r["text"].strip()]
        raw_texts = [r["text"] for r in valid_line_results]
        cleaned_texts = normalize_line_endings(raw_texts)
        for idx, cl_txt in enumerate(cleaned_texts):
            valid_line_results[idx]["text"] = cl_txt
        line_results = valid_line_results

        page_text = "\n".join(r["text"] for r in line_results if r["text"].strip())
        if page_text.strip():
            page_transcripts.append(page_text)

        # 4. Optional Character-level EMNIST review
        review_items, conf_metrics = [], {}
        if run_emnist_review and characters:
            with st.spinner("⚡ Running EMNIST CNN Character Classification & Confidence Review..."):
                emnist_str, predictions = recognize_segments(
                    emnist_model, characters, emnist_mapping, boxes, line_ids
                )
                review_items, conf_metrics = generate_confidence_review(predictions, threshold=0.75)

        # 5. Inspection & Debugging Telemetry
        if show_debug:
            tab_lines, tab_preproc, tab_emnist = st.tabs([
                "🔍 DETECTED LINE CROPS & TrOCR PREDICTIONS",
                "⚙️ PREPROCESSING & SKEW TELEMETRY",
                "🎯 EMNIST CHARACTER CLASSIFICATION"
            ])

            with tab_lines:
                st.caption(f"Detected {len(line_results)} distinct text lines for page {page_idx}. Showing line crops and TrOCR predictions:")
                for res in line_results:
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        # Display RGB line crop
                        rgb_crop = cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB)
                        st.image(rgb_crop, caption=f"Line {res['line_id'] + 1} Region: {res['region']}", width="stretch")
                    with c2:
                        st.markdown(f"""
                        <div class="line-debug-card">
                            <div class="line-badge">LINE {res['line_id'] + 1} // TrOCR PREDICTION</div>
                            <div class="line-prediction-text">{res['text'] or '(No text recognized)'}</div>
                        </div>
                        """, unsafe_allow_html=True)

            with tab_preproc:
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="1. Raw Input Feed", width="stretch")
                with pcol2:
                    gray_norm = remove_shadows(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img)
                    st.image(gray_norm, caption="2. Shadow Flattened", width="stretch")
                with pcol3:
                    st.image(clean_binary, caption="3. Clean Binary (Ruled Lines Removed)", width="stretch")
                with pcol4:
                    st.image(cv2.cvtColor(clean_color, cv2.COLOR_BGR2RGB), caption="4. Clean Color Stream (Deskewed)", width="stretch")

                st.info(f"📊 **Telemetry:** Deskew Angle: `{meta['skew_angle']:.1f}°` | Ruled Line Pixels Suppressed: `{meta['lines_removed_px']}` | Ruled Paper Detected: `{meta['has_ruled_lines']}`")

            with tab_emnist:
                if characters:
                    annotated = draw_boxes(clean_color.copy(), boxes)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Segmented Character Bounding Boxes ({len(boxes)} boxes)", width="stretch")
                    if conf_metrics:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Average EMNIST Confidence", conf_metrics.get("average_confidence_pct", "0%"))
                        m2.metric("Total Characters Segmented", conf_metrics.get("total_count", 0))
                        m3.metric("Low Confidence Characters (<75%)", conf_metrics.get("flagged_count", 0))

                        if review_items:
                            with st.expander(f"⚠️ View {len(review_items)} Low-Confidence Characters & Alternative Predictions"):
                                review_data = []
                                for item in review_items:
                                    review_data.append({
                                        "Char #": item["char_index"] + 1,
                                        "Predicted": item["predicted"],
                                        "Confidence": item["confidence_pct"],
                                        "Top Alternatives": ", ".join(item["alternatives"])
                                    })
                                st.dataframe(review_data, width="stretch")
                else:
                    st.info("No isolated characters available for EMNIST classification.")

    # ==============================
    # MODULE 02 // DIGITAL TEXT MATRIX
    # ==============================
    st.markdown("---")
    st.markdown("""
    <div class="cockpit-section-title">
        <span>02 //</span> DIGITAL TEXT MATRIX
    </div>
    """, unsafe_allow_html=True)

    combined_text = "\n\n".join(t for t in page_transcripts if t.strip())

    if combined_text.strip():
        st.success("⚡ HANDWRITING RECONSTRUCTION COMPLETE")
        edited_text = st.text_area(
            "REVIEW AND EDIT RECONSTRUCTED TEXT MATRIX:",
            value=combined_text,
            height=320
        )
    else:
        st.warning("⚠️ RECONSTRUCTION EMPTY. INPUT TEXT MANUALLY:")
        edited_text = st.text_area(
            "DIGITAL TEXT MATRIX:",
            value="",
            height=320
        )

    # ==============================
    # MODULE 03 // PDF EXPORT SYSTEM
    # ==============================
    st.markdown("---")
    st.markdown("""
    <div class="cockpit-section-title">
        <span>03 //</span> PDF EXPORT SYSTEM
    </div>
    """, unsafe_allow_html=True)

    if edited_text.strip():
        pdf_bytes = generate_pdf(edited_text)
        st.download_button(
            "⚡ GENERATE & DOWNLOAD DIGITAL PDF",
            data=pdf_bytes,
            file_name="NoteVision_Digital_Notes.pdf",
            mime="application/pdf",
            width="stretch"
        )
    else:
        st.info("ENTER TEXT ABOVE TO INITIALIZE PDF EXPORT SYSTEM.")