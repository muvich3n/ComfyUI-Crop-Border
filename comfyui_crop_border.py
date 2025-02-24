import torch
import numpy as np
from PIL import Image

class CropImageBorder:
    """
    ComfyUI node that detects and crops white/black borders from images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_border"
    CATEGORY = "image/preprocessing"

    def _detect_borders(self, img: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
        """
        Detect white/black borders in the image
        Returns: (top, right, bottom, left) border sizes
        """
        if len(img.shape) == 4:  # Handle batch dimension
            img = img[0]
            
        # Convert to numpy and ensure correct format
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            
        if img.shape[0] == 3:  # If channels first
            img = np.transpose(img, (1, 2, 0))
            
        height, width = img.shape[:2]
        
        # Convert to grayscale for easier detection
        gray = np.mean(img, axis=2)
        
        # Detect white (1.0) or black (0.0) borders
        is_white_border = np.abs(gray - 1.0) < (1 - threshold)
        is_black_border = np.abs(gray - 0.0) < threshold
        is_border = np.logical_or(is_white_border, is_black_border)
        
        # Find borders
        top = 0
        bottom = height - 1
        left = 0
        right = width - 1
        
        # Top border
        while top < height and np.all(is_border[top, :]):
            top += 1
            
        # Bottom border
        while bottom > top and np.all(is_border[bottom, :]):
            bottom -= 1
            
        # Left border
        while left < width and np.all(is_border[:, left]):
            left += 1
            
        # Right border
        while right > left and np.all(is_border[:, right]):
            right -= 1
            
        return top, right + 1, bottom + 1, left

    def crop_border(self, image: torch.Tensor, threshold: float) -> tuple[torch.Tensor]:
        """Crop white/black borders from the image"""
        try:
            # Ensure image is a proper tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image)
            
            # Convert to float32 if needed
            if image.dtype != torch.float32:
                image = image.float()
            
            # Normalize to 0-1 range if needed
            if image.max() > 1.0:
                image = image / 255.0
                
            # Convert to numpy for border detection
            image_np = image.cpu().numpy()
            
            # Detect borders
            top, right, bottom, left = self._detect_borders(image_np, threshold)
            
            # If no significant borders detected, return original image
            if top == 0 and left == 0 and bottom == image_np.shape[1] and right == image_np.shape[2]:
                return (image,)
            
            # Crop the image
            if len(image.shape) == 4:  # With batch dimension
                cropped = image[:, :, top:bottom, left:right]
            else:  # Without batch dimension
                cropped = image[:, top:bottom, left:right]
                
            # Ensure output is float32
            cropped = cropped.float()
            
            return (cropped,)
            
        except Exception as e:
            raise RuntimeError(f"Error cropping borders: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "CropImageBorder": CropImageBorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageBorder": "Crop Image Borders"
}
