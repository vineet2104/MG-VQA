"""
models.utils — Shared image encoding utility.
"""

import base64
import numpy as np
from io import BytesIO
from PIL import Image


def encode_image_to_base64(image) -> str:
    """Encode an image (ndarray, PIL, or file path) to base64 JPEG string."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Unknown image type: {type(image)}")
