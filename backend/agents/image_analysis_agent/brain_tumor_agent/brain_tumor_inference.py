import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pathlib
from typing import Dict, Any, Tuple

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BrainTumorInference:
    def __init__(self, model_path: str = 'models/brain_tumor_model.pth'):
        """
        Initialize the brain tumor inference model.
        
        Args:
            model_path: Path to the trained model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BrainTumorModel()
        
        # Resolve the model path relative to this file's directory
        if not os.path.isabs(model_path):
            current_dir = pathlib.Path(__file__).parent.absolute()
            model_path = os.path.join(current_dir, model_path)
        
        # Load the trained model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match the model's expected input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels for 4 classes in correct order
        self.classes = ['glioma', 'meningioma', 'notumo', 'pituitarytumor']

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess the input image for model inference.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Make prediction on the input brain MRI image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            input_tensor = self.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare results
            result = {
                'prediction': self.classes[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    class_name: prob.item() 
                    for class_name, prob in zip(self.classes, probabilities[0])
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def analyze_mri(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze brain MRI image and return detailed results.
        This is the main method that should be called by the agent system.
        
        Args:
            image_path: Path to the input MRI image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Make prediction
            prediction_result = self.predict(image_path)
            
            # Prepare detailed analysis
            analysis = {
                'has_tumor': prediction_result['prediction'] != 'notumo',
                'tumor_type': prediction_result['prediction'] if prediction_result['prediction'] != 'notumo' else None,
                'confidence': prediction_result['confidence'],
                'class_probabilities': prediction_result['probabilities'],
                'recommendation': self._generate_recommendation(prediction_result)
            }
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'has_tumor': None,
                'tumor_type': None,
                'confidence': 0.0,
                'class_probabilities': {class_name: 0.0 for class_name in self.classes},
                'recommendation': 'Error analyzing image. Please try again with a different image.'
            }

    def _generate_recommendation(self, prediction_result: Dict[str, Any]) -> str:
        """
        Generate a recommendation based on the prediction results.
        
        Args:
            prediction_result: Dictionary containing prediction results
            
        Returns:
            Recommendation string
        """
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        if prediction == 'notumo':
            if confidence > 0.9:
                return "High confidence of no tumor detected. Regular check-ups are still recommended."
            elif confidence > 0.7:
                return "Moderate confidence of no tumor detected. Regular check-ups are recommended."
            else:
                return "Low confidence in the analysis. Consider consulting a medical professional for a second opinion."
        else:
            tumor_type = prediction
            if confidence > 0.9:
                return f"High confidence detection of {tumor_type}. Immediate medical consultation is strongly recommended for proper diagnosis and treatment planning."
            elif confidence > 0.7:
                return f"Moderate confidence detection of {tumor_type}. Medical consultation is recommended for further evaluation."
            else:
                return f"Low confidence detection of {tumor_type}. Further medical evaluation is strongly suggested for proper diagnosis."