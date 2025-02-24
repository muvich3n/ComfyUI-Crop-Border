import torch

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
                    "default": 0.02,
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_border"
    CATEGORY = "image/preprocessing"

    def _check_row(self, img: torch.Tensor, row: int, threshold: float, target: float) -> bool:
        """Check if a row matches target color (0 for black, 1 for white)"""
        row_data = img[:, row, :]  # Get the entire row [C, W]
        diff = torch.abs(row_data - target)
        return bool(torch.all(diff < threshold).item())

    def _check_col(self, img: torch.Tensor, col: int, threshold: float, target: float) -> bool:
        """Check if a column matches target color (0 for black, 1 for white)"""
        col_data = img[:, :, col]  # Get the entire column [C, H]
        diff = torch.abs(col_data - target)
        return bool(torch.all(diff < threshold).item())

    def _detect_borders(self, img: torch.Tensor, threshold: float) -> tuple[int, int, int, int]:
        """
        Detect white/black borders in the image
        Returns: (top, left, bottom, right) border coordinates
        """
        C, H, W = img.shape
        
        # Sample the corners to determine if we're looking for white or black borders
        corner_mean = float(torch.mean(img[:, :10, :10]).item())
        is_white = corner_mean > 0.5
        target = 1.0 if is_white else 0.0
        
        print(f"Corner mean: {corner_mean:.3f}, Target: {target}")
        
        # Find borders
        top = 0
        bottom = H - 1
        left = 0
        right = W - 1
        
        # Top border
        while top < H and self._check_row(img, top, threshold, target):
            top += 1
            if top >= H - 1:  # Safety check
                top = 0
                break
                
        # Bottom border
        while bottom > top and self._check_row(img, bottom, threshold, target):
            bottom -= 1
            if bottom <= top:  # Safety check
                bottom = H - 1
                break
                
        # Left border
        while left < W and self._check_col(img, left, threshold, target):
            left += 1
            if left >= W - 1:  # Safety check
                left = 0
                break
                
        # Right border
        while right > left and self._check_col(img, right, threshold, target):
            right -= 1
            if right <= left:  # Safety check
                right = W - 1
                break
        
        # Print debug info
        print(f"Border color: {'white' if is_white else 'black'}")
        print(f"Image shape: {img.shape}")
        print(f"Detected borders - Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}")
        
        # Validate borders
        if top >= bottom or left >= right or \
           top < 0 or bottom >= H or left < 0 or right >= W or \
           (top == 0 and bottom == H-1 and left == 0 and right == W-1):
            print("Invalid borders detected")
            return 0, 0, H, W
        
        return top, left, bottom, right

    def crop_border(self, image: torch.Tensor, threshold: float) -> tuple[torch.Tensor]:
        """Crop white/black borders from the image"""
        try:
            # Handle batch dimension if present
            if len(image.shape) == 4:
                image = image[0]  # Take first image from batch
            
            # Convert HWC to CHW if needed
            if image.shape[-1] == 3:  # If last dimension is 3, it's in HWC format
                image = image.permute(2, 0, 1)  # Convert to CHW
            
            # Get original dimensions
            C, H, W = image.shape
            print(f"Original shape: {image.shape}")
            print(f"Value range: {image.min():.3f} to {image.max():.3f}")
            
            # Detect borders
            top, left, bottom, right = self._detect_borders(image, threshold)
            
            # Check if borders were actually detected
            if top == 0 and left == 0 and bottom == H and right == W:
                print("No borders detected")
                # Convert back to original format before adding batch dimension
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = image.permute(1, 2, 0)  # Convert back to HWC
                return (image.unsqueeze(0),)
            
            # Ensure we're not cropping the entire image
            if bottom - top < 10 or right - left < 10:
                print("Crop area too small")
                # Convert back to original format before adding batch dimension
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = image.permute(1, 2, 0)  # Convert back to HWC
                return (image.unsqueeze(0),)
            
            # Crop the image
            print(f"Cropping to: {top}:{bottom}, {left}:{right}")
            cropped = image[:, top:bottom, left:right].contiguous()
            print(f"Cropped shape: {cropped.shape}")
            
            # Convert back to original format
            if len(cropped.shape) == 3 and cropped.shape[0] == 3:
                cropped = cropped.permute(1, 2, 0)  # Convert back to HWC
            
            # Add batch dimension back
            cropped = cropped.unsqueeze(0)
            
            return (cropped,)
            
        except Exception as e:
            print(f"Error in crop_border: {str(e)}")
            # Return original image with batch dimension
            return (image if len(image.shape) == 4 else image.unsqueeze(0),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "CropImageBorder": CropImageBorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageBorder": "Crop Image Borders"
}
