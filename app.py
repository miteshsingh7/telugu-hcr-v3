import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import streamlit as st

# Patch streamlit.elements.image for streamlit-drawable-canvas background_image compatibility in Streamlit 1.35+
try:
    import streamlit.elements.image as sei
    import streamlit.elements.lib.image_utils as iu

    class _CanvasLayoutWrapper:
        def __init__(self, width):
            self.width = width

    def _patched_canvas_image_to_url(image, width_or_layout=None, clamp=False, channels="RGB", output_format="PNG", image_id=""):
        if isinstance(width_or_layout, int):
            layout = _CanvasLayoutWrapper(width_or_layout)
        else:
            layout = width_or_layout
        return iu.image_to_url(image, layout, clamp, channels, output_format, image_id)

    sei.image_to_url = _patched_canvas_image_to_url
except Exception:
    pass

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
from src.data.preprocess import numpy_canonical_preprocess

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
        font-size: 5.2rem;
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
    .guide-banner {
        background-color: #F0F9FF;
        border: 1px dashed #0284C7;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Telugu HCR Neural Network...")
def load_hcr_model():
    import tensorflow as tf
    
    model_paths = [
        ROOT_DIR / "checkpoints/multitask_mobilenet_best.keras",
        ROOT_DIR / "checkpoints/hierarchical_best.keras",
        ROOT_DIR / "checkpoints/telugu_v3_best.keras",
        ROOT_DIR / "checkpoints/track_b_best.keras",
        ROOT_DIR / "checkpoints/track_b_custom_cnn/best_model/model.keras",
        ROOT_DIR / "checkpoints/track_a_best.keras",
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
                try:
                    import zipfile, io
                    with zipfile.ZipFile(str(p), "r") as z:
                        cfg = json.loads(z.read("config.json"))
                    def clean_cfg(d):
                        if isinstance(d, dict):
                            d.pop("quantization_config", None)
                            for k, v in d.items():
                                clean_cfg(v)
                        elif isinstance(d, list):
                            for item in d:
                                clean_cfg(item)
                    clean_cfg(cfg)
                    buf = io.BytesIO()
                    with zipfile.ZipFile(str(p), "r") as z_in:
                        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z_out:
                            for item in z_in.infolist():
                                if item.filename == "config.json":
                                    z_out.writestr("config.json", json.dumps(cfg))
                                else:
                                    z_out.writestr(item, z_in.read(item.filename))
                    with open(str(p), "wb") as f:
                        f.write(buf.getvalue())
                    loaded_model = tf.keras.models.load_model(str(p), compile=False)
                    loaded_path = str(p)
                    break
                except Exception:
                    continue
                
    # If not found locally, attempt Hugging Face Hub download
    if loaded_model is None:
        try:
            from huggingface_hub import hf_hub_download
            hf_path = hf_hub_download(
                repo_id="miteshsingh7/telugu-hcr-v3-models",
                filename="telugu_v3_best.keras",
                local_dir=str(ROOT_DIR / "checkpoints"),
            )
            loaded_model = tf.keras.models.load_model(hf_path, compile=False)
            loaded_path = hf_path
        except Exception:
            pass

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


def make_tracing_background(glyph_text: str, size: int = 300) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    if not glyph_text:
        return img
        
    draw = ImageDraw.Draw(img)
    font_paths = [
        "/System/Library/Fonts/KohinoorTelugu.ttc",
        "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Telugu MN.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 170)
                break
            except Exception:
                continue
                
    if font:
        draw.text((size // 2, size // 2), glyph_text, font=font, fill=(235, 238, 244), anchor="mm")
    return img


def predict_character(model, tensor_in: np.ndarray):
    expected_dim = model.input_shape[1:3]
    actual_dim = tensor_in.shape[1:3]
    assert actual_dim == expected_dim, f"Resolution Mismatch: Model expects {expected_dim}, but got {actual_dim}"
    
    if len(model.input_shape) == 4 and model.input_shape[-1] == 3 and tensor_in.shape[-1] == 1:
        tensor_in = np.repeat(tensor_in, 3, axis=-1)
    elif len(model.input_shape) == 4 and model.input_shape[-1] == 1 and tensor_in.shape[-1] == 3:
        tensor_in = tensor_in[:, :, :, :1]
        
    with tf.device("/CPU:0"):
        t_tensor = tf.convert_to_tensor(tensor_in, dtype=tf.float32)
        raw_out = model(t_tensor, training=False)
        if isinstance(raw_out, list) and len(raw_out) == 3:
            # Multi-Task outputs: (base, mod, vattu)
            preds = {
                "base": raw_out[0].numpy()[0],
                "modifier": raw_out[1].numpy()[0],
                "vattu": raw_out[2].numpy()[0],
            }
        else:
            preds = raw_out.numpy()[0]
    return preds


def get_base_letter(class_name: str) -> Tuple[str, str]:
    parts = class_name.replace("/", "__").split("__")
    cat = parts[0].lower()
    if cat == "achulu":
        v = parts[1].lower() if len(parts) > 1 else "a"
        return VOWELS.get(v, "అ"), f"Vowel '{v}'"
    elif cat == "guninthamulu":
        c = parts[1].lower() if len(parts) > 1 else "ka"
        if c == "kha": return "క", "Consonant 'ka'"
        elif c == "khh": return "ఖ", "Consonant 'kha'"
        elif c == "ch": return "ఛ", "Consonant 'chha'"
        elif c == "th": return "ఠ", "Consonant 'tha'"
        elif c == "dh": return "ఢ", "Consonant 'dha'"
        elif c == "sh": return "శ", "Consonant 'sha'"
        elif c == "sha": return "ష", "Consonant 'ssha'"
        elif c == "rr": return "ఱ", "Consonant 'rra'"
        else: return CONSONANTS.get(parts[1], CONSONANTS.get(c, "క")), f"Consonant '{c}'"
    else:
        c = parts[1] if len(parts) > 1 else "ka"
        return CONSONANTS.get(c, CONSONANTS.get(c.lower(), "క")), f"Consonant '{c}'"


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
    if model is not None:
        st.success(f"**Model Loaded:** `{Path(model_path).name}`")
        st.info(f"**Input Shape:** `{model.input_shape}`")
        st.info(f"**Total Parameters:** `{model.count_params():,}`")
        st.info(f"**Classes:** `{len(class_names):,}` Telugu Categories")
    else:
        st.error("🚨 **NO MODEL LOADED!** Checkpoint not found in `checkpoints/`. Real predictions are disabled.")
        
    st.markdown("---")
    st.markdown("### 📊 Benchmark Metrics")
    st.markdown("""
    - **Top-1 Accuracy:** `74.60%` (Baseline Test Set)
    - **Top-3 Accuracy:** `93.60%` (Top-3 Candidates)
    - **Top-5 Accuracy:** `97.20%` (Near-Perfect)
    - **Classes:** `630` Categories
    """)
    st.markdown("---")
    st.markdown("Developed with **TensorFlow / Keras & Streamlit**")

if model is None:
    st.error("🚨 **No Trained Model Checkpoint Found!** Please ensure a valid `.keras` file exists in `checkpoints/` (e.g. `telugu_v3_best.keras` or `track_b_best.keras`). Inference is disabled until a model is present.")

tab_draw, tab_gallery, tab_upload, tab_explorer, tab_metrics = st.tabs([
    "🎨 Draw Character", "🖼️ Visual Sample Gallery", "📁 Upload Image", "📖 630 Character Explorer", "📊 Architecture & Metrics"
])

# ----------------- TAB 1: DRAW -----------------
with tab_draw:
    col_canvas, col_results = st.columns([1.1, 1.2])
    
    with col_canvas:
        st.markdown("#### 🖌️ Draw a Telugu Character:")
        
        c_opt1, c_opt2 = st.columns([1, 1])
        with c_opt1:
            pen_width = st.slider("Pen Thickness:", min_value=3, max_value=20, value=8, step=1)
        with c_opt2:
            trace_choice = st.selectbox(
                "Tracing Template (Overlay on Canvas):",
                ["None (Blank Canvas)", "క (ka)", "అ (a)", "ఆ (aa)", "ల (la)", "ర (ra)", "ప (pa)", "మ (ma)", "ణ (ana)", "చ (cha)", "ట (ta)"]
            )
            
        bg_image = None
        bg_color = "#FFFFFF"
        if trace_choice != "None (Blank Canvas)":
            char_to_draw = trace_choice.split(" ")[0]
            bg_image = make_tracing_background(char_to_draw, size=300)
            bg_color = ""
            st.markdown(f"""
            <div class="guide-banner">
                ✏️ Trace over the faint <b>{char_to_draw}</b> outline in the canvas below!
            </div>
            """, unsafe_allow_html=True)
            
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=pen_width,
                stroke_color="#000000",
                background_color=bg_color,
                background_image=bg_image,
                height=300,
                width=300,
                drawing_mode="freedraw",
                key=f"canvas_pad_v_{trace_choice.replace(' ', '_')}",
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
            
            if np.any(raw_data[:, :, :3] < 180):
                if current_hash != st.session_state.get("last_canvas_hash"):
                    st.session_state["last_canvas_hash"] = current_hash
                    st.session_state["sample_image_path"] = None
                
                if st.session_state.get("sample_image_path") is None:
                    has_drawing = True
                    input_image = Image.fromarray(raw_data.astype("uint8"))
                    st.caption("🎨 Source: Live Canvas Drawing")

        if input_image is None and st.session_state.get("sample_image_path") and Path(st.session_state["sample_image_path"]).exists():
            input_image = Image.open(st.session_state["sample_image_path"])
            has_drawing = True
            st.caption(f"📁 Source: Dataset Sample (`{Path(st.session_state['sample_image_path']).name}`)")
                
        if has_drawing and input_image is not None:
            if model is None:
                st.error("🚨 Inference unavailable: No trained neural network is loaded.")
            else:
                img_size = model.input_shape[1] if model.input_shape[1] is not None else 96
                num_channels = model.input_shape[-1] if len(model.input_shape) == 4 else 1
                norm_mode = "imagenet" if num_channels == 3 else "rescale"
                tensor_in, preproc_img = numpy_canonical_preprocess(
                    input_image,
                    img_size=img_size,
                    num_channels=num_channels,
                    normalize_mode=norm_mode,
                )
                
                preds = predict_character(model, tensor_in)
                
                if isinstance(preds, dict):
                    # Multi-Task Hierarchical Model Output
                    base_p = preds["base"]
                    mod_p = preds["modifier"]
                    vattu_p = preds["vattu"]
                    
                    # Load grapheme mappings
                    maps_path = Path("outputs/grapheme_maps.json")
                    if maps_path.exists():
                        with open(maps_path, "r", encoding="utf-8") as f:
                            g_maps = json.load(f)
                        base_letters = g_maps["base_letters"]
                        vowel_mods = g_maps["vowel_modifiers"]
                    else:
                        from src.data.decomposition import BASE_LETTERS, VOWEL_MODIFIERS
                        base_letters = BASE_LETTERS
                        vowel_mods = VOWEL_MODIFIERS
                    
                    top_b_idx = int(np.argmax(base_p))
                    top_b_glyph = base_letters[top_b_idx] if top_b_idx < len(base_letters) else "క"
                    top_b_conf = float(base_p[top_b_idx])
                    
                    top_m_idx = int(np.argmax(mod_p))
                    top_m_name = vowel_mods[top_m_idx] if top_m_idx < len(vowel_mods) else "none"
                    top_m_conf = float(mod_p[top_m_idx])
                    
                    # Top 3 base candidates
                    top_b_ranks = np.argsort(base_p)[::-1][:3]
                    
                    st.markdown(f"""
                    <div class="glyph-box">
                        <div style="font-size: 1.1rem; color: #166534; font-weight: 600; text-transform: uppercase;">Identified Telugu Letter (Multi-Task):</div>
                        <div class="telugu-glyph">{top_b_glyph}</div>
                        <div class="glyph-name">Base Akshara: {top_b_glyph} ({top_b_conf*100:.1f}% Confidence)</div>
                        <div class="glyph-category">Vowel Sign: {top_m_name} ({top_m_conf*100:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 🏆 Primary Letter Candidates (Top-3):")
                    for rank, b_i in enumerate(top_b_ranks, 1):
                        b_glyph = base_letters[b_i] if b_i < len(base_letters) else "?"
                        b_c = float(base_p[b_i])
                        col_b1, col_b2 = st.columns([1.5, 3.5])
                        with col_b1:
                            st.markdown(f"**#{rank} &nbsp; `{b_glyph}`**")
                        with col_b2:
                            st.progress(float(min(1.0, b_c)), text=f"{b_c*100:.1f}%")
                            
                    with st.expander("🔬 Multi-Task Head Probabilities"):
                        st.write(f"**Base Head Top-3:** {[(base_letters[i], f'{base_p[i]*100:.1f}%') for i in top_b_ranks]}")
                        top_m_ranks = np.argsort(mod_p)[::-1][:3]
                        st.write(f"**Modifier Head Top-3:** {[(vowel_mods[i], f'{mod_p[i]*100:.1f}%') for i in top_m_ranks]}")
                else:
                    # 1. Primary Base Letter Aggregation (Hierarchical Root Letter)
                    root_scores = {}
                    root_best_subclass = {}
                    for i, p in enumerate(preds):
                        bg, bd = get_base_letter(class_names[i])
                        root_scores[bg] = root_scores.get(bg, 0.0) + float(p)
                        if bg not in root_best_subclass or float(p) > root_best_subclass[bg][1]:
                            root_best_subclass[bg] = (class_names[i], float(p))
                        
                    sorted_roots = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_base_glyph, top_base_conf = sorted_roots[0]
                    
                    # 2. Winning Root's Best Diacritic Subclass
                    best_sub_cls, best_sub_prob = root_best_subclass[top_base_glyph]
                    best_glyph, best_desc, best_cat = map_class_to_telugu(best_sub_cls)
                    
                    # 3. Global Top Candidates for Detailed Debug View
                    top_indices = np.argsort(preds)[::-1][:5]
                    top1_cls = class_names[top_indices[0]]
                    
                    st.markdown(f"""
                    <div class="glyph-box">
                        <div style="font-size: 1.1rem; color: #166534; font-weight: 600; text-transform: uppercase;">Identified Telugu Character:</div>
                        <div class="telugu-glyph">{best_glyph}</div>
                        <div class="glyph-name">Root Letter: {top_base_glyph} ({best_desc}) • {top_base_conf*100:.1f}% Confidence</div>
                        <div class="glyph-category">Category: {best_cat}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 🏆 Primary Letter Candidates (Top-3):")
                    for rank, (bglyph, bconf) in enumerate(sorted_roots, 1):
                        sub_c, _ = root_best_subclass[bglyph]
                        sub_g, sub_d, _ = map_class_to_telugu(sub_c)
                        col_b1, col_b2 = st.columns([1.5, 3.5])
                        with col_b1:
                            st.markdown(f"**#{rank} &nbsp; `{sub_g}`** ({bglyph})")
                        with col_b2:
                            st.progress(float(min(1.0, bconf)), text=f"{bconf*100:.1f}%")
                            
                    with st.expander("🔬 Priority 2 Debug View (Exact Array & Logits)"):
                        col_d1, col_d2 = st.columns([1, 2])
                        with col_d1:
                            st.image(preproc_img, caption=f"Canonical Shape: {tensor_in.shape}", width=120)
                        with col_d2:
                            st.code(f"Model Input Shape: {model.input_shape}\nTop-1 Class Index: {top_indices[0]}\nClass Name: {top1_cls}\nRaw Probability: {preds[top_indices[0]]:.6f}")
                            
                    with st.expander("🔬 View Detailed 630-Class Diacritic Predictions"):
                        for rank, idx in enumerate(top_indices, 1):
                            c_name = class_names[idx]
                            glyph, desc, _ = map_class_to_telugu(c_name)
                            conf = preds[idx]
                            st.write(f"#{rank} `[Index {idx}] {glyph}` ({desc}) — {conf*100:.1f}%")
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
            if model is None:
                st.error("🚨 Inference unavailable: No trained neural network is loaded.")
            else:
                img_size = model.input_shape[1] if model.input_shape[1] is not None else 96
                tensor_up, preproc_up = numpy_canonical_preprocess(up_img, img_size=img_size, num_channels=1)
                up_preds = predict_character(model, tensor_up)
                
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
        st.markdown("""<div class="metric-card"><div class="metric-value">74.60%</div><div class="metric-label">Top-1 Accuracy (Baseline)</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-value">93.60%</div><div class="metric-label">Top-3 Accuracy</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-value">97.20%</div><div class="metric-label">Top-5 Accuracy</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card"><div class="metric-value">630</div><div class="metric-label">Classes Supported</div></div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    ### 🔬 Pipeline Specifications:
    - **Dataset Size:** 292,752 handwritten character images across 630 unique classes.
    - **Input Representation:** 96×96 Grayscale canonical normalized tensors.
    - **Model Architecture:** Custom 6-Block CNN with Batch Normalization, Dropout, and Global Average Pooling.
    - **Optimization:** AdamW optimizer with Warmup Cosine Learning Rate Decay.
    """)
