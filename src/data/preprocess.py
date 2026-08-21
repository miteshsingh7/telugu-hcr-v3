"""Canonical preprocessing module for Telugu Handwritten Character Recognition.

Provides dual-mode canonical preprocessing:
1. numpy_canonical_preprocess: For live web-serving, single-image inference in Streamlit/OpenCV.
2. tf_canonical_preprocess: Native C++ TensorFlow ops for GIL-free, high-throughput training in tf.data.
"""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def numpy_canonical_preprocess(
    img_input: Union[Image.Image, np.ndarray],
    img_size: int = 96,
    num_channels: int = 1,
    fill_ratio: float = 0.55,
) -> Tuple[np.ndarray, Image.Image]:
    """NumPy / OpenCV canonical preprocessor for live inference.
    
    1. Grayscale conversion and alpha channel blending with white background.
    2. Background color normalization (ensures white background ~245-255, dark ink).
    3. Content bounding box extraction & centering with natural 55% frame fill matching training data.
    4. Anti-aliased area resizing and ink density calibration.
    5. Float32 normalization in [0.0, 1.0].
    
    Returns:
        tensor: np.ndarray of shape (1, img_size, img_size, num_channels), float32 in [0, 1].
        display_img: PIL Image of the preprocessed image.
    """
    if isinstance(img_input, Image.Image):
        if img_input.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img_input.size, (255, 255, 255))
            bg.paste(img_input, mask=img_input.split()[-1])
            img_gray = np.array(bg.convert("L"))
        else:
            img_gray = np.array(img_input.convert("L"))
    else:
        if img_input.ndim == 3:
            if img_input.shape[2] == 4:
                alpha = img_input[:, :, 3].astype(float) / 255.0
                rgb = img_input[:, :, :3].astype(float)
                white_bg = np.ones_like(rgb) * 255.0
                comp = rgb * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])
                img_gray = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY) if _HAS_CV2 else np.mean(comp, axis=2).astype(np.uint8)
            elif img_input.shape[2] == 3:
                img_gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY) if _HAS_CV2 else np.mean(img_input, axis=2).astype(np.uint8)
            else:
                img_gray = img_input[:, :, 0]
        else:
            img_gray = img_input.copy()

    # Ensure white background (~245-255) and dark ink
    if np.mean(img_gray) < 127:
        img_gray = 255 - img_gray

    h, w = img_gray.shape[:2]
    
    # If image is a large canvas drawing (e.g. 300x300 canvas), crop to bounding box and center at 55% fill
    if max(h, w) > 120 and _HAS_CV2:
        ink_mask = (img_gray < 210).astype(np.uint8) * 255
        if np.any(ink_mask > 0):
            coords = cv2.findNonZero(ink_mask)
            bx, by, bw, bh = cv2.boundingRect(coords)
            cropped = img_gray[by : by + bh, bx : bx + bw]

            target_len = int(img_size * fill_ratio)
            scale = target_len / max(bw, bh)
            nw = max(2, int(bw * scale))
            nh = max(2, int(bh * scale))

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(cropped, (nw, nh), interpolation=interp)

            # Soften high-contrast digital black (0) to natural ballpoint ink density (~110-160)
            sim_ink = np.clip(246 - (255 - resized.astype(float)) * 0.65, 0, 255).astype(np.uint8)

            canvas = np.full((img_size, img_size), 246, dtype=np.uint8)
            off_x = (img_size - nw) // 2
            off_y = (img_size - nh) // 2
            canvas[off_y : off_y + nh, off_x : off_x + nw] = sim_ink
        else:
            canvas = np.full((img_size, img_size), 246, dtype=np.uint8)
    else:
        # Standard dataset scan: pad to square and resize
        if h != w and _HAS_CV2:
            max_d = max(h, w)
            top = (max_d - h) // 2
            bottom = max_d - h - top
            left = (max_d - w) // 2
            right = max_d - w - left
            padded = cv2.copyMakeBorder(img_gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        else:
            padded = img_gray

        if _HAS_CV2:
            interp = cv2.INTER_AREA if max(h, w) >= img_size else cv2.INTER_LINEAR
            canvas = cv2.resize(padded, (img_size, img_size), interpolation=interp)
        else:
            canvas = np.array(Image.fromarray(padded).resize((img_size, img_size)))

    arr_norm = canvas.astype(np.float32) / 255.0
    arr_norm = np.clip(arr_norm, 0.0, 1.0)

    if num_channels == 1:
        tensor = np.expand_dims(np.expand_dims(arr_norm, -1), 0)
    elif num_channels == 3:
        arr_3ch = np.repeat(arr_norm[..., np.newaxis], 3, axis=-1)
        tensor = np.expand_dims(arr_3ch, 0)
    else:
        tensor = np.expand_dims(np.expand_dims(arr_norm, -1), 0)

    display_img = Image.fromarray(canvas)
    return tensor, display_img


def tf_canonical_preprocess(
    img_raw_or_tensor: Union[tf.Tensor, bytes],
    img_size: int = 96,
    num_channels: int = 1,
    normalize_mode: str = "rescale",
) -> tf.Tensor:
    """Pure TensorFlow C++ canonical preprocessor for tf.data pipelines.
    
    Runs entirely outside the Python GIL with full multithreaded efficiency.
    """
    if isinstance(img_raw_or_tensor, tf.Tensor) and img_raw_or_tensor.dtype == tf.string:
        img = tf.io.decode_image(img_raw_or_tensor, channels=3, expand_animations=False)
    elif isinstance(img_raw_or_tensor, tf.Tensor):
        img = img_raw_or_tensor
        if tf.shape(img)[-1] == 1:
            img = tf.repeat(img, 3, axis=-1)
    else:
        img = tf.io.decode_image(img_raw_or_tensor, channels=3, expand_animations=False)

    img = tf.image.rgb_to_grayscale(img[:, :, :3])
    img.set_shape([None, None, 1])

    # Ensure white background (255) and dark ink (0)
    mean_val = tf.reduce_mean(tf.cast(img, tf.float32))
    img = tf.cond(mean_val < 127.0, lambda: 255 - img, lambda: img)

    # Pad to square to preserve aspect ratio
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    max_dim = tf.maximum(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = tf.pad(img, [[top, bottom], [left, right], [0, 0]], mode="CONSTANT", constant_values=255)

    # Resize using bilinear for upscaling and area for downscaling
    img_resized = tf.cond(
        max_dim > img_size,
        lambda: tf.image.resize(img_padded, [img_size, img_size], method="area"),
        lambda: tf.image.resize(img_padded, [img_size, img_size], method="bilinear"),
    )

    # Clamp to valid [0, 255] range
    img_resized = tf.clip_by_value(img_resized, 0.0, 255.0)

    if num_channels == 3:
        img_resized = tf.repeat(img_resized, 3, axis=-1)

    img_resized = tf.cast(img_resized, tf.float32)

    if normalize_mode == "imagenet" and num_channels == 3:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        img_normalized = img_resized - mean
    else:
        img_normalized = img_resized / 255.0

    # Final safety clamp to [0, 1]
    img_normalized = tf.clip_by_value(img_normalized, 0.0, 1.0)

    img_normalized.set_shape([img_size, img_size, num_channels])
    return img_normalized
