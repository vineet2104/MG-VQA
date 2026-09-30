import torch
import os
import numpy as np
from torchvision.ops import nms
from ultralytics import SAM

class SAM3Segmentator:
    """
    Wrapper class for SAM 3 inference and Open-World Grounding filtering.
    """
    def __init__(self, model_weights: str = "sam3.pt"):
        here = os.path.dirname(os.path.abspath(__file__))

        if model_weights is None:
            model_weights = os.path.join(here, "sam3.pt")
        elif not os.path.isabs(model_weights):
            model_weights = os.path.join(here, model_weights)

        if not os.path.exists(model_weights):
            raise FileNotFoundError(
                f"SAM3 weights not found at: {model_weights}\n"
                f"(sam.py directory: {here})"
            )

        print(f"Loading SAM 3 model from {model_weights}...")
        self.model = SAM(model_weights)
        print("SAM 3 loaded successfully.")
 
    def predict(self, 
                image_np: np.ndarray, 
                conf: float = 0.25, 
                half: bool = False,
                verbose: bool = False):
        """
        Runs the image through SAM 3, applies OWG filters, and extracts
        the boolean masks and their corresponding SAM IDs.

        Args:
            image_np: Raw RGB image array (H, W, 3) from the camera.
            verbose: Whether to print Ultralytics inference logs.

        Returns:
            dict: Contains:
                - 'sam_masks': numpy array (N, H, W) of boolean masks.
                - 'sam_ids': 1D numpy array of integers [1, 2, ..., N].
                - 'raw_result': The sliced Ultralytics Result object for debugging/visualization.
        """
        # Run raw inference
        results = self.model(
            image_np, 
            conf=conf, 
            half=half, 
            verbose=verbose
        )
        raw_result = results[0]  # Extract single image result

        # Apply filters to remove a background mask and nested maskss
        filtered_result = filter_sam_masks(raw_result)

        # Extract masks into NumPy format
        if filtered_result.masks is None or len(filtered_result.masks.data) == 0:
            # Handle empty scene gracefully
            H, W = image_np.shape[:2]
            return {
                'sam_masks': np.empty((0, H, W), dtype=bool),
                'sam_ids': np.empty((0,), dtype=int),
                'raw_result': filtered_result
            }

        # Convert to boolean numpy array (N, H, W)
        sam_masks_np = filtered_result.masks.data.bool().cpu().numpy()
        N = sam_masks_np.shape[0]
        
        # Generate arbitrary tracking IDs (1 through N)
        sam_ids_np = np.arange(1, N + 1, dtype=int)

        return {
            'sam_masks': sam_masks_np,
            'sam_ids': sam_ids_np,
            'raw_result': filtered_result
        }
    
def filter_sam_masks(
    result,
    nms_iou=0.5,
    inside_ratio=0.95,
    box_ioa_gate=0.90,        
    max_area_frac=0.70,
    border_touch_frac=0.15
):
    """
    Combined OWG-style filtering:
      1. NMS on boxes.
      2. Background removal (area + border-touch) executed FIRST.
      3. Nested-mask removal with a tolerant Bounding Box IoA gate.
    """
    # Error handler: No masks found
    if result is None or result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return result

    # Extract to CPU for memory safety
    masks = result.masks.data.bool().cpu()            # (N, H, W)
    boxes = result.boxes.xyxy.float().cpu()           # (N, 4)
    scores = result.boxes.conf.float().cpu()          # (N,)

    N, H, W = masks.shape
    if N == 0:
        return result

    # Non-Maximum Suppression (NMS) on bounding boxes
    keep_nms = nms(boxes, scores, nms_iou)
    masks = masks[keep_nms]
    boxes = boxes[keep_nms]
    scores = scores[keep_nms]

    # Sort by confidence score descending (highest score first)
    order = torch.argsort(scores, descending=True)
    masks = masks[order]
    boxes = boxes[order]
    scores = scores[order]

    # Background Mask Filtering (Vectorized)
    N_current = masks.shape[0]
    
    # Precompute areas with a clamp to avoid division by zero
    areas = masks.flatten(1).sum(dim=1).float().clamp(min=1.0) 
    area_frac = areas / float(H * W)

    # Calculate border touch ratios 
    top_touch = masks[:, 0, :].float().mean(dim=1)
    bottom_touch = masks[:, H - 1, :].float().mean(dim=1)
    left_touch = masks[:, :, 0].float().mean(dim=1)
    right_touch = masks[:, :, W - 1].float().mean(dim=1)
    
    any_border_touch = torch.maximum(
        torch.maximum(top_touch, bottom_touch),
        torch.maximum(left_touch, right_touch)
    )

    # Identify masks that are NOT backgrounds
    keep_bg = (area_frac <= max_area_frac) & (any_border_touch <= border_touch_frac)
    keep_bg_idx = torch.nonzero(keep_bg).squeeze(1)

    # If everything was considered background, keep the highest scoring mask
    if keep_bg_idx.numel() == 0 and N_current > 0:
        # Fallback: keep the best-scoring one if everything was a background
        keep_bg_idx = torch.tensor([0], dtype=torch.long)

    # Filter variables down to just the foreground objects
    masks = masks[keep_bg_idx]
    boxes = boxes[keep_bg_idx]
    areas = areas[keep_bg_idx]

    # Nested Mask Removal (With Bounding Box IoA Gate)
    keep_local = []
    
    for j in range(len(masks)):
        aj = areas[j].item()
        if aj <= 0:
            continue

        x1j, y1j, x2j, y2j = boxes[j]
        is_nested = False

        for i in keep_local:
            x1i, y1i, x2i, y2i = boxes[i]

            # Tolerant Gate: Bounding Box IoA
            ix1 = max(x1i.item(), x1j.item())
            iy1 = max(y1i.item(), y1j.item())
            ix2 = min(x2i.item(), x2j.item())
            iy2 = min(y2i.item(), y2j.item())
            
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area_box = iw * ih
            
            area_j_box = max(1.0, (x2j.item() - x1j.item()) * (y2j.item() - y1j.item()))
            box_ioa = inter_area_box / area_j_box
            
            if box_ioa < box_ioa_gate:
                continue

            # Exact Check: Pixel-level containment
            inter_pixels = (masks[i] & masks[j]).sum().item()
            if (inter_pixels / aj) >= inside_ratio:
                is_nested = True
                break

        if not is_nested:
            keep_local.append(j)

    if not keep_local:
        return result[[]]

    # Map Indices Back to Original Result Object
    keep_local_tensor = torch.tensor(keep_local, dtype=torch.long)
    final_keep_idx = keep_nms[order[keep_bg_idx[keep_local_tensor]]]

    return result[final_keep_idx]