import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import streamlit as st

# Permanent thread-safe monkey-patch for Keras 3.11 in multi-threaded web servers
try:
    import keras.src.backend.common.name_scope as ns
    import keras.src.backend.common.global_state as gs

    def _safe_name_scope_exit(self, *args, **kwargs):
        if getattr(self, "_pop_on_exit", False):
            stack = gs.get_global_attribute("name_scope_stack")
            if stack:
                stack.pop()

    ns.name_scope.__exit__ = _safe_name_scope_exit
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.telugu_unicode import map_class_to_telugu, CONSONANTS, VOWELS

st.set_page_config(
    page_title="Telugu Handwritten Character Recognizer (v3)",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .glyph-box {
        background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
        border: 2px solid #86EFAC;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .telugu-glyph {
        font-size: 5rem;
        font-weight: bold;
        color: #15803D;
        line-height: 1.1;
    }
    .glyph-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 6px;
    }
    .glyph-category {
        font-size: 0.95rem;
        color: #4B5563;
        background-color: #E5E7EB;
        padding: 3px 12px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 6px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
    .gallery-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_hcr_model():
    import tensorflow as tf
    
    model_paths = [
        ROOT_DIR / "checkpoints/telugu_v3_best.keras",
        ROOT_DIR / "checkpoints/track_a_best.keras",
        ROOT_DIR / "checkpoints/track_b_best.keras",
    ]
    
    loaded_model = None
    loaded_path = None
    for p in model_paths:
        if p.exists():
            try:
                loaded_model = tf.keras.models.load_model(str(p), compile=False)
                loaded_path = str(p)
                break
            except Exception:
                continue
                
    class_names = []
    class_map_paths = [
        ROOT_DIR / "outputs/class_names.json",
        ROOT_DIR / "data_samples/class_names.json",
    ]
    
    for cmp in class_map_paths:
        if cmp.exists():
            with open(cmp, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    class_names = data
                    break
                elif isinstance(data, dict):
                    class_names = [k for k, _ in sorted(data.items(), key=lambda x: x[1])]
                    break
                    
    if not class_names:
        class_names = [f"Class_{i}" for i in range(630)]
        
    return loaded_model, loaded_path, class_names


def preprocess_image(img: Image.Image, img_size: int = 96) -> Tuple[np.ndarray, Image.Image]:
    gray = img.convert("L")
    arr = np.array(gray)
    
    if np.mean(arr) < 127:
        arr = 255 - arr
        
    ink_mask = (arr < 220).astype(np.uint8) * 255
    
    if np.any(ink_mask > 0):
        coords = cv2.findNonZero(ink_mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = ink_mask[y:y+h, x:x+w]
        
        target_size = int(img_size * 0.70)
        scale = target_size / max(w, h)
        new_w = max(2, int(w * scale))
        new_h = max(2, int(h * scale))
        
        resized_ink = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_ink = cv2.dilate(resized_ink, kernel, iterations=1)
        blurred_ink = cv2.GaussianBlur(dilated_ink, (3, 3), 0.8)
        
        final_canvas = np.full((img_size, img_size), 255, dtype=np.uint8)
        off_x = (img_size - new_w) // 2
        off_y = (img_size - new_h) // 2
        final_canvas[off_y:off_y+new_h, off_x:off_x+new_w] = 255 - blurred_ink
    else:
        final_canvas = np.full((img_size, img_size), 255, dtype=np.uint8)
        
    arr_norm = final_canvas.astype(np.float32) / 255.0
    tensor = np.expand_dims(np.expand_dims(arr_norm, -1), 0)
    return tensor, Image.fromarray(final_canvas)


def predict_character(model, tensor_in: np.ndarray) -> np.ndarray:
    if len(model.input_shape) == 4 and model.input_shape[-1] == 3:
        tensor_in = np.repeat(tensor_in, 3, axis=-1)
    preds = model(tensor_in, training=False).numpy()[0]
    return preds


def extract_root_grapheme(class_name: str) -> Tuple[str, str]:
    parts = class_name.replace("/", "__").split("__")
    category = parts[0].lower()
    if category == "achulu":
        v = parts[1] if len(parts) > 1 else "a"
        glyph = VOWELS.get(v.lower(), "అ")
        return glyph, f"Vowel '{v}'"
    else:
        c = parts[1] if len(parts) > 1 else "ka"
        glyph = CONSONANTS.get(c, CONSONANTS.get(c.lower(), "క"))
        return glyph, f"Consonant '{c}'"


model, model_path, class_names = load_hcr_model()

if "sample_image_path" not in st.session_state:
    st.session_state["sample_image_path"] = None

if "last_canvas_hash" not in st.session_state:
    st.session_state["last_canvas_hash"] = ""

st.markdown('<div class="main-header">✍️ Telugu Handwritten Character Recognizer (v3)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Character Recognition for 630 Telugu Aksharas (Achulu, Hallulu, Guninthamulu, Othulu)</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Telugu_alphabet.png/320px-Telugu_alphabet.png", width="stretch")
    st.markdown("### ⚙️ Model Information")
    if model:
        st.success(f"**Model Loaded:** `{Path(model_path).name}`")
        st.info(f"**Total Parameters:** `{model.count_params():,}`")
        st.info(f"**Classes:** `{len(class_names):,}` Telugu Characters")
    else:
        st.warning("Running in Demo Mode.")
        
    st.markdown("---")
    st.markdown("### 📊 Benchmark Metrics")
    st.markdown("""
    - **Top-1 Accuracy:** `85.64%` (Fine-Tuned v3)
    - **Top-3 Accuracy:** `97.11%` (Top-3 Candidates)
    - **Top-5 Accuracy:** `98.64%` (Near-Perfect)
    - **Classes:** `630` Categories
    """)
    st.markdown("---")
    st.markdown("Developed with **TensorFlow / Keras & Streamlit**")

tab_draw, tab_gallery, tab_upload, tab_explorer, tab_metrics = st.tabs([
    "🎨 Draw Character", "🖼️ Visual Sample Gallery", "📁 Upload Image", "📖 630 Character Explorer", "📊 Architecture & Metrics"
])

# ----------------- TAB 1: DRAW -----------------
with tab_draw:
    col_canvas, col_results = st.columns([1.1, 1.2])
    
    with col_canvas:
        st.markdown("#### 🖌️ Draw a Telugu Character:")
        pen_width = st.slider("Pen Thickness:", min_value=3, max_value=16, value=6, step=1)
            
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=pen_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="telugu_main_canvas",
            )
            has_canvas = True
        except ImportError:
            has_canvas = False
            st.warning("Install `streamlit-drawable-canvas` for interactive drawing.")
            
        if st.session_state.get("sample_image_path"):
            st.info(f"Viewing Sample: `{Path(st.session_state['sample_image_path']).name}`")
            if st.button("🧹 Clear Sample / Switch Back to Canvas", use_container_width=True):
                st.session_state["sample_image_path"] = None
                st.rerun()

    with col_results:
        st.markdown("#### 🎯 Recognition Results:")
        
        has_drawing = False
        input_image = None
        
        if has_canvas and canvas_result is not None and canvas_result.image_data is not None:
            raw_data = canvas_result.image_data
            current_hash = hashlib.md5(raw_data.tobytes()).hexdigest()
            
            if np.mean(raw_data[:, :, :3]) < 252 or np.any(raw_data[:, :, :3] < 120):
                if current_hash != st.session_state.get("last_canvas_hash"):
                    st.session_state["last_canvas_hash"] = current_hash
                    st.session_state["sample_image_path"] = None
                
                if st.session_state.get("sample_image_path") is None:
                    has_drawing = True
                    input_image = Image.fromarray(raw_data.astype("uint8")).convert("RGB")
                    st.caption("🎨 Source: Live Canvas Drawing")

        if input_image is None and st.session_state.get("sample_image_path") and Path(st.session_state["sample_image_path"]).exists():
            input_image = Image.open(st.session_state["sample_image_path"])
            has_drawing = True
            st.caption(f"📁 Source: Dataset Sample (`{Path(st.session_state['sample_image_path']).name}`)")
                
        if has_drawing and input_image is not None:
            tensor_in, preproc_img = preprocess_image(input_image, img_size=96)
            
            if model is not None:
                preds = predict_character(model, tensor_in)
            else:
                preds = np.zeros(len(class_names))
                preds[0] = 0.88
                preds[1] = 0.08
                preds[2] = 0.04
                
            top_indices = np.argsort(preds)[::-1][:5]
            top1_cls = class_names[top_indices[0]]
            top1_glyph, top1_desc, top1_cat = map_class_to_telugu(top1_cls)
            top1_conf = preds[top_indices[0]] * 100
            
            # Root Family Calculation
            root_glyph, root_desc = extract_root_grapheme(top1_cls)
            root_conf = sum(preds[i] for i in top_indices if extract_root_grapheme(class_names[i])[0] == root_glyph) * 100
            
            st.markdown(f"""
            <div class="glyph-box">
                <div class="telugu-glyph">{top1_glyph}</div>
                <div class="glyph-name">{top1_desc}</div>
                <div class="glyph-category">{top1_cat} • Exact: {top1_conf:.1f}% | Root Family: {root_conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 🏆 Top Candidate Glyphs:")
            for rank, idx in enumerate(top_indices, 1):
                c_name = class_names[idx]
                glyph, desc, cat = map_class_to_telugu(c_name)
                conf = preds[idx]
                
                col_r1, col_r2 = st.columns([1.2, 3.8])
                with col_r1:
                    st.markdown(f"**#{rank} &nbsp; `{glyph}`** ({desc})")
                with col_r2:
                    st.progress(float(min(1.0, conf)), text=f"{conf*100:.1f}%")
                    
            with st.expander("🔍 Preprocessed 96×96 Input View"):
                st.image(preproc_img, caption="Normalized Input fed into Neural Network", width=120)
        else:
            st.info("Draw a character on the canvas on the left or select a sample from the Visual Gallery!")


# ----------------- TAB 2: VISUAL SAMPLE GALLERY -----------------
with tab_gallery:
    st.markdown("#### 🖼️ Visual Character Image Gallery (Click any image to test):")
    st.caption("Browse actual handwritten images from the dataset. Click 'Test this Character' on any card to evaluate it instantly!")
    
    gallery_dir = ROOT_DIR / "data_samples/sample_gallery"
    if gallery_dir.exists():
        gallery_images = sorted(list(gallery_dir.glob("*.*")))
        
        cols = st.columns(4)
        for i, img_path in enumerate(gallery_images):
            col = cols[i % 4]
            cls_name = img_path.stem
            glyph, desc, cat = map_class_to_telugu(cls_name)
            
            with col:
                st.markdown(f"""
                <div class="gallery-card">
                    <div style="font-size: 2.2rem; font-weight: bold; color: #1E3A8A;">{glyph}</div>
                    <div style="font-size: 0.85rem; color: #4B5563;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(str(img_path), width=100)
                if st.button(f"🚀 Test `{glyph}`", key=f"btn_gal_{i}", use_container_width=True):
                    st.session_state["sample_image_path"] = str(img_path)
                    st.session_state["last_canvas_hash"] = ""
                    st.toast(f"Loaded {glyph} ({desc})! Switched to Draw tab.")
                    st.rerun()


# ----------------- TAB 3: UPLOAD -----------------
with tab_upload:
    st.markdown("#### 📁 Upload Handwritten Telugu Image:")
    uploaded_file = st.file_uploader("Upload a handwritten image (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        up_img = Image.open(uploaded_file)
        col_u1, col_u2 = st.columns([1, 1.2])
        
        with col_u1:
            st.image(up_img, caption="Uploaded Image", width=250)
            
        with col_u2:
            tensor_up, preproc_up = preprocess_image(up_img, img_size=96)
            if model is not None:
                up_preds = predict_character(model, tensor_up)
            else:
                up_preds = np.zeros(len(class_names))
                up_preds[0] = 0.91
                up_preds[1] = 0.05
                
            top_up_indices = np.argsort(up_preds)[::-1][:5]
            up_top1_cls = class_names[top_up_indices[0]]
            up_glyph, up_desc, up_cat = map_class_to_telugu(up_top1_cls)
            up_conf = up_preds[top_up_indices[0]] * 100
            
            st.markdown(f"""
            <div class="glyph-box">
                <div class="telugu-glyph">{up_glyph}</div>
                <div class="glyph-name">{up_desc}</div>
                <div class="glyph-category">{up_cat} • Confidence: {up_conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Top Predictions:")
            for rank, idx in enumerate(top_up_indices, 1):
                c_name = class_names[idx]
                glyph, desc, _ = map_class_to_telugu(c_name)
                conf = up_preds[idx]
                st.progress(float(min(1.0, conf)), text=f"#{rank} {glyph} ({desc}) — {conf*100:.1f}%")


# ----------------- TAB 4: EXPLORER -----------------
with tab_explorer:
    st.markdown("#### 📖 630 Telugu Character Database Explorer:")
    st.caption("Browse through the complete Telugu handwriting vocabulary supported by this model.")
    
    search_q = st.text_input("🔍 Search by Romanized Name or Telugu Glyph (e.g. 'ka', 'aa', 'క')", "")
    
    explorer_items = []
    for cls in class_names:
        glyph, desc, cat = map_class_to_telugu(cls)
        if search_q.lower() in cls.lower() or search_q in glyph or search_q.lower() in desc.lower():
            explorer_items.append({"Class ID": cls, "Telugu Glyph": glyph, "Description": desc, "Category": cat})
            
    st.dataframe(explorer_items, width="stretch", height=450)


# ----------------- TAB 5: METRICS -----------------
with tab_metrics:
    st.markdown("#### 📊 System Architecture & Performance Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card"><div class="metric-value">85.64%</div><div class="metric-label">Top-1 Accuracy (v3)</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-value">97.11%</div><div class="metric-label">Top-3 Accuracy</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-value">98.64%</div><div class="metric-label">Top-5 Accuracy</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card"><div class="metric-value">630</div><div class="metric-label">Classes Supported</div></div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    ### 🔬 Pipeline Specifications:
    - **Dataset Size:** 292,752 handwritten character images across 630 unique classes.
    - **Input Representation:** 96×96 Grayscale normalized tensors with dynamic augmentation.
    - **Model Architecture:** Custom 80-Layer ResNet CNN with Residual Conv Blocks, Batch Normalization, and Dropout.
    - **Optimization:** AdamW optimizer with Warmup Cosine Learning Rate Decay and Label Smoothing (`0.05`).
    """)
