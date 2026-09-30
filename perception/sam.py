"""
perception.sam — SAM3 segmentation model loading and inference.

Functions:
  - load_sam3_model: Load SAM3 weights
  - re_perceive_object: Re-segment using GT centroid
  - build_sam_occupancy: Combined binary mask for push ray-tracing
"""

from tools import (
    load_sam3_model,
    re_perceive_object,
    build_sam_occupancy,
)

__all__ = ["load_sam3_model", "re_perceive_object", "build_sam_occupancy"]
