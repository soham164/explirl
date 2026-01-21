"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Explainer
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union
from PIL import Image

from .image_explainer import ImageExplainer
from ..utils.image_utils import apply_heatmap, normalize_heatmap, deprocess_image


class GradCAMExplainer(ImageExplainer):
    """
    Grad-CAM explainer for CNN models
    
    Generates heatmaps showing which regions of an image are important
    for the model's prediction.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[Union[str, torch.nn.Module]] = None,
        target_size: tuple = (224, 224),
        device: Optional[str] = None
    ):
        """
        Initialize Grad-CAM explainer
        
        Args:
            model: PyTorch CNN model
            target_layer: Layer to compute gradients from (name or module)
            target_size: Target image size (height, width)
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__(model, target_size, device)
        
        # Find target layer
        if target_layer is None:
            # Try to find last convolutional layer
            self.target_layer = self._find_last_conv_layer()
        elif isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self) -> torch.nn.Module:
        """Find the last convolutional layer in the model"""
        conv_layers = []
        
        def find_conv(module, prefix=''):
            for name, layer in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(layer, torch.nn.Conv2d):
                    conv_layers.append((full_name, layer))
                else:
                    find_conv(layer, full_name)
        
        find_conv(self.model)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in model")
        
        print(f"Using layer: {conv_layers[-1][0]}")
        return conv_layers[-1][1]
    
    def _get_layer_by_name(self, layer_name: str) -> torch.nn.Module:
        """Get layer by its name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_class: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation
        
        Args:
            image: Input image
            target_class: Target class index (if None, uses predicted class)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - heatmap: Grad-CAM heatmap (H, W)
            - predicted_class: Predicted class index
            - target_class: Target class used for explanation
            - confidence: Prediction confidence
            - image: Original image
        """
        # Preprocess image
        image_tensor = self._validate_instance(image)
        image_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get prediction info
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0, target_class].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy and normalize
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = normalize_heatmap(heatmap)
        
        # Store original image for visualization
        if isinstance(image, str):
            from ..utils.image_utils import load_image
            original_image = load_image(image, self.target_size)
        elif isinstance(image, Image.Image):
            original_image = np.array(image.resize((self.target_size[1], self.target_size[0])))
        elif isinstance(image, np.ndarray):
            original_image = image
        else:
            original_image = deprocess_image(image_tensor)
        
        self.explanation = {
            'heatmap': heatmap,
            'predicted_class': output.argmax(dim=1).item(),
            'target_class': target_class,
            'confidence': confidence,
            'image': original_image,
            'probabilities': probabilities.cpu().detach().numpy()[0]
        }
        
        return self.explanation
    
    def visualize(
        self,
        explanation: Optional[Dict] = None,
        alpha: float = 0.5,
        colormap: str = 'jet',
        figsize: tuple = (12, 4),
        **kwargs
    ):
        """
        Visualize Grad-CAM explanation
        
        Args:
            explanation: Explanation dictionary from explain()
            alpha: Transparency of heatmap overlay (0-1)
            colormap: Matplotlib colormap name
            figsize: Figure size
            **kwargs: Additional matplotlib parameters
        """
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        import cv2
        
        # Get data
        image = explanation['image']
        heatmap = explanation['heatmap']
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply heatmap overlay
        overlayed = apply_heatmap(image, heatmap, alpha=alpha)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap=colormap)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title(
            f"Overlay (Class: {explanation['target_class']}, "
            f"Conf: {explanation['confidence']:.2%})"
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
