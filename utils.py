import cv2
import numpy as np
import base64
import io
from typing import List, Dict
from PIL import Image

def masks_to_polygons(mask: np.ndarray, min_area: int = 64) -> List[List[List[float]]]:
    """Convert a binary mask (H,W) 0/255 to list of polygons (list of [x,y])."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pts = cnt.reshape(-1, 2).astype(float).tolist()
        polys.append(pts)
    # Sort by area desc so first is biggest
    polys = sorted(polys, key=lambda p: cv2.contourArea(np.array(p, dtype=np.float32)), reverse=True)
    return polys

def draw_instances(bgr_img: np.ndarray, instances: List[Dict]) -> np.ndarray:
    out = bgr_img.copy()
    for inst in instances:
        box = inst.get("box", None)
        poly = inst.get("polygon", None)
        score = inst.get("score", 0.0)
        label = inst.get("label", "defect")
        if box:
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, f"{label}:{score:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        if poly and poly.get("points"):
            pts = np.array(poly["points"], dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(out, [pts], isClosed=True, color=(255,0,0), thickness=2)
            cv2.fillPoly(out, [pts], color=(255,0,0,))
            # Add transparency by blending
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color=(255,0,0))
            out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
    return out

def img_to_base64(bgr_img: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
