# Backend (FastAPI + PyTorch Mask R-CNN)

## Setup
```bash
# create & activate a virtualenv (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Note: This will install PyTorch/torchvision. If you need specific CUDA builds,
> follow https://pytorch.org/get-started/locally/ to install matching torch/torchvision,
> then remove those lines from requirements and re-run `pip -r requirements.txt` for the rest.

## Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /predict` — send an image; returns detected defect polygons, boxes, scores, and a quick annotated preview (base64).
- `GET /health` — health check.

## How it works
- Loads a **Mask R-CNN** (`torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")`).
- Treats all detected instances as *potential defects*. For a custom *defect* class, fine-tune on your own dataset.
- Converts binary masks to **polygon contours** for lightweight UI overlay.
