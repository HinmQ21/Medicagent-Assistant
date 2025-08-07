import os
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_model_checkpoint(model_name, local_path):
    """Download model checkpoint from various sources."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Model download URLs - these can be updated as needed
    model_urls = {
        'bone_fracture_yolov8': {
            # This is the trained model from RuiyangJu's repository
            'url': 'https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8/releases/download/Trained_model/best.pt',
            'description': 'YOLOv8 trained on GRAZPEDWRI-DX bone fracture dataset'
        },
        'yolov8n_pretrained': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt',
            'description': 'YOLOv8n pretrained on COCO dataset'
        }
    }
    
    if model_name not in model_urls:
        logger.warning(f"Model {model_name} not found in predefined URLs")
        return False
    
    try:
        model_info = model_urls[model_name]
        url = model_info['url']
        
        logger.info(f"Downloading {model_info['description']} from {url}")
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Show progress for large files
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (1024 * 1024) == 0:  # Show progress every MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Model downloaded successfully to {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        
        # Clean up partial download
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return False

def verify_model_file(model_path):
    """Verify that the model file exists and is valid."""
    if not os.path.exists(model_path):
        return False
    
    # Check file size (should be > 1MB for valid model)
    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        logger.warning(f"Model file {model_path} seems too small ({file_size} bytes)")
        return False
    
    logger.info(f"Model file verified: {model_path} ({file_size} bytes)")
    return True

def download_bone_fracture_model(model_path):
    """Download bone fracture detection model specifically."""
    logger.info("Downloading bone fracture detection model...")
    
    # Try to download the trained bone fracture model
    success = download_model_checkpoint('bone_fracture_yolov8', model_path)
    
    if success and verify_model_file(model_path):
        logger.info("Bone fracture model downloaded and verified successfully")
        return True
    else:
        logger.warning("Failed to download bone fracture model, will use pretrained YOLOv8")
        return False

if __name__ == "__main__":
    # Test download
    test_path = "./models/test_bone_fracture.pt"
    download_bone_fracture_model(test_path)