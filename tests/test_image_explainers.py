"""
Tests for image explainers
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SimpleCNN(nn.Module):
    """Simple CNN for testing"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple CNN model for testing"""
    model = SimpleCNN(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


class TestGradCAMExplainer:
    """Test suite for Grad-CAM explainer"""
    
    def test_initialization(self, simple_model):
        """Test that Grad-CAM explainer initializes correctly"""
        from expliRL import GradCAMExplainer
        
        explainer = GradCAMExplainer(simple_model)
        assert explainer.model is simple_model
        assert explainer.target_layer is not None
    
    def test_explain_with_image(self, simple_model, sample_image):
        """Test explanation generation with image"""
        from expliRL import GradCAMExplainer
        
        explainer = GradCAMExplainer(simple_model, target_layer=simple_model.features[-2])
        explanation = explainer.explain(sample_image)
        
        assert 'heatmap' in explanation
        assert 'predicted_class' in explanation
        assert 'confidence' in explanation
        assert explanation['heatmap'].shape == (224, 224)
        assert 0 <= explanation['confidence'] <= 1
    
    def test_explain_with_array(self, simple_model):
        """Test explanation with numpy array"""
        from expliRL import GradCAMExplainer
        
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        explainer = GradCAMExplainer(simple_model, target_layer=simple_model.features[-2])
        explanation = explainer.explain(img_array)
        
        assert 'heatmap' in explanation
        assert explanation['heatmap'].shape == (224, 224)
    
    def test_target_class_specification(self, simple_model, sample_image):
        """Test specifying target class"""
        from expliRL import GradCAMExplainer
        
        explainer = GradCAMExplainer(simple_model, target_layer=simple_model.features[-2])
        explanation = explainer.explain(sample_image, target_class=5)
        
        assert explanation['target_class'] == 5


class TestLIMEImageExplainer:
    """Test suite for LIME image explainer"""
    
    def test_initialization(self, simple_model):
        """Test that LIME explainer initializes correctly"""
        from expliRL import LIMEImageExplainer
        
        explainer = LIMEImageExplainer(simple_model)
        assert explainer.model is simple_model
    
    def test_explain_with_image(self, simple_model, sample_image):
        """Test explanation generation"""
        from expliRL import LIMEImageExplainer
        
        explainer = LIMEImageExplainer(simple_model)
        explanation = explainer.explain(
            sample_image,
            num_samples=100,  # Reduced for faster testing
            num_features=5,
            n_segments=20
        )
        
        assert 'explanation_mask' in explanation
        assert 'top_features' in explanation
        assert 'segments' in explanation
        assert len(explanation['top_features']) <= 5
    
    def test_segmentation_methods(self, simple_model, sample_image):
        """Test different segmentation methods"""
        from expliRL import LIMEImageExplainer
        
        explainer = LIMEImageExplainer(simple_model)
        
        for method in ['quickshift', 'slic']:
            explanation = explainer.explain(
                sample_image,
                num_samples=50,
                segmentation_method=method,
                n_segments=20
            )
            assert 'segments' in explanation


class TestImageUtils:
    """Test suite for image utility functions"""
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        from expliRL.utils.image_utils import preprocess_image
        
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = preprocess_image(img_array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_deprocess_image(self):
        """Test image deprocessing"""
        from expliRL.utils.image_utils import preprocess_image, deprocess_image
        
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = preprocess_image(img_array)
        recovered = deprocess_image(tensor)
        
        assert recovered.shape == (224, 224, 3)
        assert recovered.dtype == np.uint8
    
    def test_apply_heatmap(self):
        """Test heatmap application"""
        from expliRL.utils.image_utils import apply_heatmap
        
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224)
        
        overlayed = apply_heatmap(image, heatmap)
        
        assert overlayed.shape == image.shape
        assert overlayed.dtype == np.uint8
    
    def test_create_superpixels(self):
        """Test superpixel creation"""
        from expliRL.utils.image_utils import create_superpixels
        
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        segments = create_superpixels(image, method='quickshift')
        
        assert segments.shape == (224, 224)
        assert len(np.unique(segments)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
