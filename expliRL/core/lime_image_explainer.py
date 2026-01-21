"""
LIME (Local Interpretable Model-agnostic Explanations) for Images
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union
from PIL import Image
from sklearn.linear_model import Ridge

from .image_explainer import ImageExplainer
from ..utils.image_utils import create_superpixels, preprocess_image


class LIMEImageExplainer(ImageExplainer):
    """
    LIME explainer for image classification models
    
    Segments image into superpixels and explains which segments
    are important for the prediction.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_size: tuple = (224, 224),
        device: Optional[str] = None
    ):
        """
        Initialize LIME image explainer
        
        Args:
            model: PyTorch model for image classification
            target_size: Target image size (height, width)
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__(model, target_size, device)
        self.segments = None
    
    def explain(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_class: Optional[int] = None,
        num_samples: int = 1000,
        num_features: int = 10,
        segmentation_method: str = 'quickshift',
        n_segments: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for image
        
        Args:
            image: Input image
            target_class: Target class (if None, uses predicted class)
            num_samples: Number of perturbed samples to generate
            num_features: Number of top features to show
            segmentation_method: Method for superpixel segmentation
            n_segments: Approximate number of superpixels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing explanation
        """
        # Load and preprocess image
        if isinstance(image, str):
            from ..utils.image_utils import load_image
            original_image = load_image(image, self.target_size)
        elif isinstance(image, Image.Image):
            original_image = np.array(image.resize((self.target_size[1], self.target_size[0])))
        elif isinstance(image, np.ndarray):
            import cv2
            if image.shape[:2] != self.target_size:
                original_image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            else:
                original_image = image.copy()
        else:
            raise ValueError("Unsupported image type")
        
        # Get original prediction
        image_tensor = self._validate_instance(original_image)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        original_prob = probabilities[0, target_class].item()
        
        # Create superpixels
        self.segments = create_superpixels(
            original_image,
            method=segmentation_method,
            n_segments=n_segments
        )
        
        num_segments = len(np.unique(self.segments))
        
        # Generate perturbed samples
        X_samples = np.random.randint(0, 2, size=(num_samples, num_segments))
        y_samples = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Create perturbed image
            perturbed = self._create_perturbed_image(original_image, X_samples[i])
            
            # Get prediction
            perturbed_tensor = preprocess_image(perturbed, target_size=self.target_size)
            perturbed_tensor = perturbed_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(perturbed_tensor)
                prob = torch.softmax(output, dim=1)[0, target_class].item()
            
            y_samples[i] = prob
        
        # Fit linear model
        distances = np.sum((X_samples - 1) ** 2, axis=1)
        weights = np.exp(-distances / (0.25 * num_segments))
        
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(X_samples, y_samples, sample_weight=weights)
        
        # Get feature importance
        coefficients = linear_model.coef_
        
        # Get top features
        top_indices = np.argsort(np.abs(coefficients))[-num_features:][::-1]
        top_features = [(idx, coefficients[idx]) for idx in top_indices]
        
        # Create explanation mask
        explanation_mask = np.zeros_like(self.segments, dtype=float)
        for idx, coef in top_features:
            explanation_mask[self.segments == idx] = coef
        
        # Normalize mask
        if explanation_mask.max() > 0:
            explanation_mask = explanation_mask / np.abs(explanation_mask).max()
        
        self.explanation = {
            'image': original_image,
            'segments': self.segments,
            'explanation_mask': explanation_mask,
            'top_features': top_features,
            'predicted_class': output.argmax(dim=1).item(),
            'target_class': target_class,
            'confidence': original_prob,
            'num_segments': num_segments,
            'linear_model': linear_model
        }
        
        return self.explanation
    
    def _create_perturbed_image(self, image: np.ndarray, active_segments: np.ndarray) -> np.ndarray:
        """
        Create perturbed image by hiding inactive segments
        
        Args:
            image: Original image
            active_segments: Binary array indicating which segments to keep
            
        Returns:
            Perturbed image
        """
        perturbed = image.copy()
        
        for segment_id in range(len(active_segments)):
            if active_segments[segment_id] == 0:
                # Hide this segment (set to gray)
                mask = self.segments == segment_id
                perturbed[mask] = 128  # Gray color
        
        return perturbed
    
    def visualize(
        self,
        explanation: Optional[Dict] = None,
        positive_only: bool = True,
        figsize: tuple = (15, 5),
        **kwargs
    ):
        """
        Visualize LIME explanation
        
        Args:
            explanation: Explanation dictionary from explain()
            positive_only: Show only positive contributions
            figsize: Figure size
            **kwargs: Additional parameters
        """
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        image = explanation['image']
        mask = explanation['explanation_mask']
        
        # Create positive and negative masks
        positive_mask = np.maximum(mask, 0)
        negative_mask = np.maximum(-mask, 0)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(explanation['segments'], cmap='tab20')
        axes[1].set_title(f"Superpixels ({explanation['num_segments']})")
        axes[1].axis('off')
        
        # Positive contributions
        axes[2].imshow(image)
        axes[2].imshow(positive_mask, cmap='Greens', alpha=0.6)
        axes[2].set_title('Positive Contributions')
        axes[2].axis('off')
        
        # Negative contributions (or combined if positive_only)
        if positive_only:
            # Show only top positive segments
            top_mask = np.zeros_like(mask)
            for idx, coef in explanation['top_features'][:5]:
                if coef > 0:
                    top_mask[explanation['segments'] == idx] = 1
            
            masked_image = image.copy()
            masked_image[top_mask == 0] = masked_image[top_mask == 0] * 0.3
            axes[3].imshow(masked_image.astype(np.uint8))
            axes[3].set_title('Top Positive Regions')
        else:
            axes[3].imshow(image)
            axes[3].imshow(negative_mask, cmap='Reds', alpha=0.6)
            axes[3].set_title('Negative Contributions')
        
        axes[3].axis('off')
        
        plt.suptitle(
            f"LIME Explanation (Class: {explanation['target_class']}, "
            f"Conf: {explanation['confidence']:.2%})"
        )
        plt.tight_layout()
        plt.show()
        
        return fig
