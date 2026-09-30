"""
perception.overlays — Visual overlays, A/B markers, and debug annotations.

Functions:
  - compute_ab_marker_info: Compute A/B target centroid positions
  - format_ab_targets_block: Format A/B info as text for prompt
  - draw_ab_markers: Draw A/B labels on image (for singlepass)
  - draw_pointing_debug: Annotate pointing model output
  - save_gt_annotated_image: GT bounding-box debug image
"""

from tools import (
    compute_ab_marker_info,
    format_ab_targets_block,
    draw_ab_markers,
    draw_pointing_debug,
    save_gt_annotated_image,
    compute_centroid_rowcol,
    compute_quadrant,
)

__all__ = [
    "compute_ab_marker_info",
    "format_ab_targets_block",
    "draw_ab_markers",
    "draw_pointing_debug",
    "save_gt_annotated_image",
    "compute_centroid_rowcol",
    "compute_quadrant",
]
