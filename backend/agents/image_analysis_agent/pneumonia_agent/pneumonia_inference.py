import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision import models


PNEUMONIA_CLASSES: List[str] = [
    "normal",      # 0
    "pneumonia"    # 1
]


class PneumoniaClassifier:
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build ResNet18 model skeleton for binary classification
        self.model = models.resnet18(weights=None)
        # Modify first conv layer for grayscale (pneumonia X-rays often grayscale)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify classifier for binary classification
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, len(PNEUMONIA_CLASSES))
        self.model.to(self.device)
        self.model.eval()

        weights_file = Path(model_path)
        if weights_file.is_file():
            try:
                state = torch.load(weights_file, map_location=self.device)
                # support both state_dict or full checkpoint with 'state_dict'
                state_dict = state.get('state_dict', state) if isinstance(state, dict) else state
                # remove possible prefixes like 'model.'
                if isinstance(state_dict, dict):
                    cleaned = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}
                    missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
                    if missing:
                        self.logger.warning(f"Missing keys when loading pneumonia model: {missing}")
                    if unexpected:
                        self.logger.warning(f"Unexpected keys when loading pneumonia model: {unexpected}")
                else:
                    # Direct state_dict load
                    self.model.load_state_dict(state, strict=False)
                self.logger.info(f"Loaded pneumonia weights from {weights_file}")
            except Exception as e:
                self.logger.error(f"Failed to load pneumonia weights from {weights_file}: {e}")
        else:
            self.logger.warning(f"Pneumonia weights not found at {weights_file}. The model will output random-like predictions.")

        # Preprocessing pipeline tuned for chest X-rays
        self.transform = T.Compose([
            T.Lambda(lambda img: ImageOps.exif_transpose(img)),
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Grayscale(num_output_channels=1),  # Convert to grayscale for chest X-rays
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])  # Single channel normalization
        ])

    @torch.inference_mode()
    def predict(self, image_path: str) -> Dict:
        """Return pneumonia classification result.

        Returns dict: {
          'predicted_label': str,
          'predicted_index': int,
          'probabilities': {label: float},
          'confidence': float
        }
        """
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        top_index = int(max(range(len(probs)), key=lambda i: probs[i]))
        confidence = float(max(probs))
        
        result = {
            'predicted_label': PNEUMONIA_CLASSES[top_index],
            'predicted_index': top_index,
            'probabilities': {label: float(probs[i]) for i, label in enumerate(PNEUMONIA_CLASSES)},
            'confidence': confidence
        }
        return result

