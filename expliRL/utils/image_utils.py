"""
Image preprocessing and utility functions for Image XAI
"""

import numpy as np
import torch
from PIL import Image
import cv2
from typing import Union, Tuple, Optional


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        target_size: Optional (height, width) to resize to
        
    Returns:
        Image as numpy array (H, W, C) in RGB format
    """
    img = Image.open(image_path).convert('RGB')
    
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    return np.array(img)


def preprocess_image(
    image: Union[np.ndarray, Image.Image, str],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Preprocess image for neural network input
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Target size (height, width)
        normalize: Whether to normalize with ImageNet stats
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Preprocessed image tensor (1, C, H, W)
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image, target_size)
    elif isinstance(image, Image.Image):
        if target_size:
            image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Resize if needed
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization if requested
    if normalize:
        image = (image - np.array(mean)) / np.array(std)
    
    # Convert to tensor (C, H, W) with float32 dtype
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    
    # Add batch dimension (1, C, H, W)
    return image_tensor.unsqueeze(0)


def deprocess_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Convert normalized tensor back to displayable image
    
    Args:
        tensor: Image tensor (1, C, H, W) or (C, H, W)
        mean: Mean values used in normalization
        std: Std values used in normalization
        
    Returns:
        Image as numpy array (H, W, C) in [0, 255] range
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose to (H, W, C)
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Denormalize
    image = image * np.array(std) + np.array(mean)
    
    # Clip to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1) * 255
    
    return image.astype(np.uint8)


def apply_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Apply heatmap overlay on image
    
    Args:
        image: Original image (H, W, C)
        heatmap: Heatmap values (H, W) in [0, 1] range
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
        
    Returns:
        Image with heatmap overlay (H, W, C)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to [0, 255]
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def create_superpixels(
    image: np.ndarray,
    method: str = 'quickshift',
    n_segments: int = 100
) -> np.ndarray:
    """
    Segment image into superpixels for LIME
    
    Args:
        image: Input image (H, W, C)
        method: Segmentation method ('quickshift', 'slic', 'felzenszwalb')
        n_segments: Approximate number of segments
        
    Returns:
        Segmentation mask (H, W) with segment labels
    """
    from skimage.segmentation import quickshift, slic, felzenszwalb
    
    if method == 'quickshift':
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    elif method == 'slic':
        segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return segments


def get_image_gradients(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> torch.Tensor:
    """
    Compute gradients of model output w.r.t. input image
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, C, H, W)
        target_class: Target class index (if None, uses predicted class)
        
    Returns:
        Gradients tensor same shape as input
    """
    model.eval()
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor)
    
    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    output[0, target_class].backward()
    
    return image_tensor.grad


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize heatmap to [0, 1] range
    
    Args:
        heatmap: Input heatmap
        
    Returns:
        Normalized heatmap
    """
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target (height, width)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
