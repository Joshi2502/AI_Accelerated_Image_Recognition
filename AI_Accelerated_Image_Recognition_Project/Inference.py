# inference.py
from typing import Dict
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def prepare_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

@torch.inference_mode()
def predict_probs(model: torch.nn.Module, img: Image.Image, transform, device: torch.device, amp: bool = True) -> torch.Tensor:
    x = transform(img).unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W]
    with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    return probs
