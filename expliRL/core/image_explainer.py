"""
Base class for image explainers
"""

from abc import abstractmethod
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
from PIL import Image

from .base_explainer import BaseExplainer
from ..utils.image_utils import preprocess_image, deprocess_image


class ImageExplainer(BaseExplainer):
    """Abstract base class for image explainers"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_size: tuple = (224, 224),
        device: Optional[str] = None
    ):
        """
        Initialize image explainer
        
        Args:
            model: PyTorch model for image classification
            target_size: Target image size (height, width)
            device: Device to run model on ('cuda' or 'cpu')
        """
        super().__init__(model, data=None)
        self.target_size = target_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def _validate_instance(self, instance: Union[str, np.ndarray, Image.Image, torch.Tensor]):
        """
        Validate and preprocess image instance
        
        Args:
            instance: Input image (path, array, PIL Image, or tensor)
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(instance, torch.Tensor):
            # Already a tensor
            if instance.dim() == 3:
                instance = instance.unsqueeze(0)
            return instance.to(self.device)
        
        # Convert to tensor
        tensor = preprocess_image(instance, target_size=self.target_size)
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Get model prediction for image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        image_tensor = self._validate_instance(image)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'logits': output.cpu().numpy()[0]
        }
    
    @abstractmethod
    def explain(self, image: Union[str, np.ndarray, Image.Image], **kwargs) -> Dict:
        """
        Generate explanation for image
        
        Args:
            image: Input image
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing explanation
        """
        pass
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs):
        """
        Visualize explanation
        
        Args:
            explanation: Explanation dictionary from explain()
            **kwargs: Visualization parameters
        """
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        # Default implementation - subclasses should override
        print("Visualization not implemented for this explainer type.")
        print("Explanation keys:", explanation.keys())
