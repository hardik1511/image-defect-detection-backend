# backend/model.py
import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np

NUM_CLASSES = 2  # background + defect
WEIGHTS_PATH = os.getenv("DEFECT_WEIGHTS", "maskrcnn_defect.pth")
CONFIDENCE_THRESHOLD = float(os.getenv("DEFECT_CONF_THRESH", "0.5"))

_device = None
_model = None
_tf = transforms.Compose([
    transforms.ToTensor(),  # HWC [0..255] -> CHW [0..1]
])

def _build_model():
    # start from architecture only; we'll load your finetuned weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    # replace classifiers to match NUM_CLASSES
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model

def load_model():
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model()

    # load your fine-tuned weights
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Defect model weights not found at '{WEIGHTS_PATH}'. "
            "Place maskrcnn_defect.pth beside model.py or set DEFECT_WEIGHTS env var."
        )
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.eval().to(_device)
    _model = model
    return _model

@torch.no_grad()
def infer_instances(model, rgb_np):
    """
    rgb_np: HWC uint8 (RGB)
    returns dict with boxes, labels, scores, masks (binary uint8 arrays), filtered by CONFIDENCE_THRESHOLD
    """
    x = _tf(rgb_np).to(next(model.parameters()).device)
    pred = model([x])[0]

    # confidence filtering
    scores = pred.get("scores", torch.empty(0)).detach().cpu()
    keep = scores >= CONFIDENCE_THRESHOLD

    def _pick(key, default_empty):
        if key not in pred:
            return default_empty
        t = pred[key]
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
            if t.ndim == 0:
                return default_empty
            return t[keep] if keep.numel() else t[:0]
        return default_empty

    boxes  = _pick("boxes",  torch.zeros((0,4)))
    labels = _pick("labels", torch.zeros((0,), dtype=torch.int64))
    scores = scores[keep] if keep.numel() else torch.zeros((0,))

    masks_list = []
    if "masks" in pred and pred["masks"].numel() > 0:
        masks = pred["masks"].detach().cpu()  # [N,1,H,W]
        masks = masks[keep] if keep.numel() else masks[:0]
        for i in range(masks.shape[0]):
            m = (masks[i, 0].numpy() > 0.5).astype("uint8") * 255
            masks_list.append(m)

    return {
        "boxes":  boxes.numpy() if torch.is_tensor(boxes) else np.zeros((0,4)),
        "labels": labels.numpy() if torch.is_tensor(labels) else np.zeros((0,)),
        "scores": scores.numpy() if torch.is_tensor(scores) else np.zeros((0,)),
        "masks":  masks_list,
        "threshold": float(CONFIDENCE_THRESHOLD),
    }
