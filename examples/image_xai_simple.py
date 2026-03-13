"""
Simple Image XAI Demo (No Internet Required)
============================================
Demonstrates Grad-CAM with synthetic data
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from expliRL import GradCAMExplainer


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_synthetic_image(size=(224, 224)):
    """Create a synthetic image with patterns"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add some patterns
    # Red square in top-left
    img[20:80, 20:80, 0] = 255
    
    # Green circle in center
    center = (size[0]//2, size[1]//2)
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center[1])**2 + (y - center[0])**2 <= 40**2
    img[mask, 1] = 255
    
    # Blue diagonal
    for i in range(min(size)):
        if i < size[0] and i < size[1]:
            img[i, i, 2] = 255
    
    # Add some noise
    noise = np.random.randint(0, 50, size=(size[0], size[1], 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return img


def main():
    print("="*60)
    print("  Simple Image XAI Demo")
    print("="*60)
    
    # Create simple model
    print("\n1. Creating simple CNN model...")
    model = SimpleCNN(num_classes=10)
    model.eval()
    print("   ✓ Model created")
    
    # Create synthetic image
    print("\n2. Creating synthetic test image...")
    image_array = create_synthetic_image()
    image = Image.fromarray(image_array)
    print(f"   ✓ Image created (size: {image.size})")
    
    # Get prediction  # <-- NOTICE: This line needs to be indented
    print("\n3. Getting model prediction...")
    from expliRL.utils.image_utils import preprocess_image
    
    image_tensor = preprocess_image(image)
    # Add this line to convert to float32
    image_tensor = image_tensor.float()
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Grad-CAM Explanation
    print("\n" + "="*60)
    print("  Grad-CAM Explanation")
    print("="*60)
    
    try:
        print("\n4. Generating Grad-CAM heatmap...")
        
        # Use the last conv layer
        gradcam = GradCAMExplainer(model, target_layer=model.features[-3])
        
        gradcam_explanation = gradcam.explain(image)
        
        print(f"   ✓ Grad-CAM generated successfully")
        print(f"   Target class: {gradcam_explanation['target_class']}")
        print(f"   Confidence: {gradcam_explanation['confidence']:.2%}")
        
        print("\n5. Visualizing Grad-CAM...")
        gradcam.visualize(gradcam_explanation)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print("""
This demo showed:
- Creating a simple CNN model
- Generating synthetic test images
- Using Grad-CAM to visualize model attention
- Interpreting heatmaps to understand predictions

The heatmap shows which regions of the image the model
focuses on when making its prediction.

Try modifying:
- The synthetic image patterns
- The model architecture
- The target layer for Grad-CAM
    """)


if __name__ == "__main__":
    main()