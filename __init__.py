from .comfyui_crop_border import CropImageBorder

NODE_CLASS_MAPPINGS = {
    "CropImageBorder": CropImageBorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageBorder": "Crop Image Borders"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
