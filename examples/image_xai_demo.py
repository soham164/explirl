"""
expliRL Image XAI Demo
======================
Demonstrates Grad-CAM and LIME for image classification
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
from io import BytesIO

from expliRL import GradCAMExplainer, LIMEImageExplainer


def download_sample_image(url: str = None) -> Image.Image:
    """Download a sample image for demonstration"""
    if url is None:
        # Default: a cat image from ImageNet
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Could not download image: {e}")
        print("Creating a random image for demonstration...")
        # Create a random image as fallback
        random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(random_img)


def main():
    print("="*60)
    print("  expliRL Image XAI Demonstration")
    print("="*60)
    
    # Load pre-trained model
    print("\n1. Loading pre-trained ResNet50 model...")
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Download sample image
    print("\n2. Loading sample image...")
    try:
        image = download_sample_image()
        print("   ✓ Image loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading image: {e}")
        print("   Creating synthetic image...")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Display image info
    print(f"   Image size: {image.size}")
    
    # Get prediction
    print("\n3. Getting model prediction...")
    from expliRL.utils.image_utils import preprocess_image
    
    image_tensor = preprocess_image(image)
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
        gradcam = GradCAMExplainer(model)
        gradcam_explanation = gradcam.explain(image)
        
        print(f"   ✓ Grad-CAM generated successfully")
        print(f"   Target class: {gradcam_explanation['target_class']}")
        print(f"   Confidence: {gradcam_explanation['confidence']:.2%}")
        
        print("\n5. Visualizing Grad-CAM...")
        gradcam.visualize(gradcam_explanation)
        
    except Exception as e:
        print(f"   ✗ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
    
    # LIME Explanation
    print("\n" + "="*60)
    print("  LIME Image Explanation")
    print("="*60)
    
    try:
        print("\n6. Generating LIME explanation...")
        print("   (This may take a minute - generating 1000 samples)")
        
        lime = LIMEImageExplainer(model)
        lime_explanation = lime.explain(
            image,
            num_samples=1000,
            num_features=10,
            n_segments=50
        )
        
        print(f"   ✓ LIME generated successfully")
        print(f"   Number of superpixels: {lime_explanation['num_segments']}")
        print(f"   Top features identified: {len(lime_explanation['top_features'])}")
        
        print("\n7. Visualizing LIME explanation...")
        lime.visualize(lime_explanation)
        
    except Exception as e:
        print(f"   ✗ LIME error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60)
    print("""
What we demonstrated:

1. Grad-CAM - Shows which regions the CNN focuses on
   → Fast, works with any CNN
   → Great for understanding model attention

2. LIME - Explains predictions using superpixels
   → Model-agnostic, works with any classifier
   → Shows which image regions matter most

Next steps:
• Try with your own images
• Experiment with different models
• Adjust visualization parameters
• Use for model debugging and validation
    """)


if __name__ == "__main__":
    main()
