import os
import torch
from torch import no_grad

class RealESRGANUpsampler:
    """Real-ESRGAN super-resolution upsampler for low-resolution jersey crops"""
    
    def __init__(self, model_path=None, device=None, upscale=4):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.upscale = upscale
        self.model = None
        self.loaded = False
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model_cfg = dict(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=upscale
            )
            model = RRDBNet(**model_cfg)
            
            url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x{upscale}plus.pth'
            self.model = RealESRGANer(
                scale=upscale, model_path=model_path or url, model=model,
                tile=400, tile_pad=10, pre_pad=0, half=(self.device == 'cuda')
            )
            self.loaded = True
            
        except Exception as e:
            print(f"Warning: Real-ESRGAN load failed: {e}")
    
    def upscale_image(self, image):
        if not self.loaded or self.model is None or image is None:
            return image
        
        try:
            with no_grad():
                upscaled, _ = self.model.enhance(image, outscale=self.upscale)
            return upscaled
        except Exception:
            return image
    
    def should_upscale(self, height, threshold=64):
        return height < threshold
    
    def process_crop(self, crop_image, threshold=64):
        if crop_image is None or crop_image.size == 0:
            return crop_image, False
        
        height = crop_image.shape[0]
        if self.should_upscale(height, threshold):
            return self.upscale_image(crop_image), True
        return crop_image, False
