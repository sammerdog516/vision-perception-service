from __future__ import annotations

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from app.models.cnn import CNN

# Must match training normalization
_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

ArrayLike = Union[np.ndarray, torch.Tensor, list]

def preprocess(image):
    """
    image: (28, 28) array-like (0-255 uint8 or 0-1 float)
    returns: (1, 1, 28, 28) float tensor
    """
    if torch.is_tensor(image):
        x = image.float()

    else:
        x = _preprocess(image)
        
    if x.ndim == 2:
        x = x.unsqueeze(0) # (1, 28, 28)
    
    return x.unsqueeze(0)   # batch dim

def infer(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x: (1, 1, 28, 28)
    returns: logits (1, 10)
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
    return logits

def predict(model: torch.nn.Module, image: ArrayLike, device: torch.device) -> Tuple[int, float]:
    """
    image: raw image (28, 28)
    returns: (digit, confidence)
    """
    x = preprocess(image)
    logits = infer(model, x, device)

    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    return int(pred.item()), float(conf.item())

def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads CNN weights and returns model on device
    """
    model = CNN()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model