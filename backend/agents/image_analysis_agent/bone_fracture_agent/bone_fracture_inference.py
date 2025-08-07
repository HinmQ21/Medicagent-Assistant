import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from .model_download import download_model_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class BoneFractureDetection:
    """Handles bone fracture detection using YOLOv8 model."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = DEVICE
        self.model = self._load_model()
        
        # Class names for GRAZPEDWRI-DX dataset (9 classes)
        self.class_names = [
            'boneanomaly',      # 0
            'bonelesion',       # 1  
            'foreignbody',      # 2
            'fracture',         # 3
            'metal',            # 4
            'periostealreaction', # 5
            'pronatorsign',     # 6
            'softtissue',       # 7
            'text'              # 8
        ]
        
        # Colors for visualization (9 different colors for 9 classes)
        self.colors = [
            (255, 0, 0),     # Red - boneanomaly
            (0, 255, 0),     # Green - bonelesion  
            (0, 0, 255),     # Blue - foreignbody
            (255, 255, 0),   # Yellow - fracture (main class)
            (255, 0, 255),   # Magenta - metal
            (0, 255, 255),   # Cyan - periostealreaction
            (128, 0, 128),   # Purple - pronatorsign
            (255, 165, 0),   # Orange - softtissue
            (128, 128, 128)  # Gray - text
        ]
    
    def _load_model(self):
        """Load the trained YOLOv8 model."""
        try:
            # Check if YOLOv8 is available
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics not installed. Please install it using: pip install ultralytics")
                raise ImportError("ultralytics package required for YOLOv8")
            
            # Download model if not exists
            if not os.path.exists(self.model_path):
                logger.info(f"Model not found at {self.model_path}. Attempting to download...")
                # First try to download the trained bone fracture model
                try:
                    download_model_checkpoint('bone_fracture_yolov8', self.model_path)
                except Exception as e:
                    logger.warning(f"Could not download bone fracture model: {e}")
                    logger.info("Falling back to pretrained YOLOv8n model...")
                    # Use pretrained YOLOv8n as fallback
                    model = YOLO('yolov8n.pt')
                    model.save(self.model_path)
            
            # Load the model
            model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def _draw_predictions(self, image, results, output_path):
        """Draw bounding boxes and labels on the image."""
        try:
            # Convert image to RGB if it's BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Create figure and axes
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            # Process detection results
            detections_found = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    detections_found = True
                    
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only process if confidence is above threshold
                        if confidence > 0.25:  # 25% confidence threshold
                            # Get class name
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            # Create rectangle with class-specific color
                            color_idx = class_id if class_id < len(self.colors) else 0
                            rect = patches.Rectangle(
                                (x1, y1), x2 - x1, y2 - y1,
                                linewidth=2,
                                edgecolor=np.array(self.colors[color_idx]) / 255.0,
                                facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            # Add label with class-specific color
                            label = f"{class_name}: {confidence:.2f}"
                            ax.text(
                                x1, y1 - 10,
                                label,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=np.array(self.colors[color_idx]) / 255.0, alpha=0.7),
                                fontsize=10,
                                color='white'
                            )
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Bone Fracture Detection Results", fontsize=14, fontweight='bold')
            
            # Save the image
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            logger.info(f"Detection visualization saved to {output_path}")
            return detections_found
            
        except Exception as e:
            logger.error(f"Error drawing predictions: {e}")
            raise e
    
    def predict(self, image_path, output_path):
        """Detect bone fractures in an image and return visualization."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Run inference
            results = self.model(image_path, conf=0.25)  # 25% confidence threshold
            
            # Draw predictions
            detections_found = self._draw_predictions(image, results, output_path)
            
            # Return detection summary
            detection_count = 0
            total_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].cpu().numpy()
                        if confidence > 0.25:
                            detection_count += 1
                            total_confidence += confidence
            
            avg_confidence = total_confidence / detection_count if detection_count > 0 else 0
            
            return {
                'detections_found': detections_found,
                'detection_count': detection_count,
                'average_confidence': avg_confidence,
                'output_path': output_path
            }
            
        except Exception as e:
            logger.error(f"Error during bone fracture detection: {e}")
            raise e


# Example Usage
if __name__ == "__main__":
    detector = BoneFractureDetection(model_path="./models/bone_fracture_yolov8.pt")
    result = detector.predict("test_xray.jpg", "output.png")
    print(f"Detection results: {result}")