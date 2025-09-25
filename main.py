from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import numpy as np
import cv2
from PIL import Image
import io
import base64

from model import load_model, infer_instances
from utils import masks_to_polygons, draw_instances, img_to_base64

app = FastAPI(title="Image Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
confidence_threshold = 0.5

class Polygon(BaseModel):
    points: List[List[float]]  # [[x,y], ...]

class InstanceOut(BaseModel):
    box: List[float]           # [x1,y1,x2,y2]
    score: float
    label: str
    polygon: Polygon | None

class PredictResponse(BaseModel):
    width: int
    height: int
    instances: List[InstanceOut]
    preview_base64: str        # PNG dataURL (no header)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(img)  # HWC RGB

    outputs = infer_instances(model, np_img)  # dict with boxes, scores, labels, masks
    H, W = np_img.shape[:2]

    instances = []
    drawn = np_img.copy()[:, :, ::-1]  # to BGR for cv2 drawing
    for i in range(len(outputs["scores"])):
        score = float(outputs["scores"][i])
        if score < confidence_threshold:
            continue
        box = outputs["boxes"][i].tolist()  # [x1,y1,x2,y2]
        label_id = int(outputs["labels"][i])
        label = f"defect_{label_id}"
        poly = None

        mask = outputs["masks"][i]
        if mask is not None:
            # Convert to polygon(s)
            polygons = masks_to_polygons(mask)
            if polygons:
                poly = {"points": polygons[0]}  # take largest/first polygon

        instances.append({"box": box, "score": score, "label": label, "polygon": poly})

    # Draw preview
    if len(instances) > 0:
        drawn = draw_instances(drawn, instances)

    # encode preview as base64 PNG (without data URL header; frontend will prefix)
    preview_b64 = img_to_base64(drawn)

    return {"width": W, "height": H, "instances": instances, "preview_base64": preview_b64}
