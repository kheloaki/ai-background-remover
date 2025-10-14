#!/usr/bin/env python3
"""
Background Removal Script
Automatically removes backgrounds from images using AI-powered background removal.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import argparse
from rembg import remove, new_session
import logging
import numpy as np
import cv2
from skimage import restoration, filters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackgroundRemover:
    def __init__(self, model_name='u2net'):
        """
        Initialize the background remover with a specific model.
        
        Args:
            model_name (str): The model to use for background removal.
                             Options: 'u2net', 'u2net_human_seg', 'u2netp', 'silueta', 'isnet-general-use'
        """
        self.model_name = model_name
        self.session = new_session(model_name)
        logger.info(f"Initialized background remover with model: {model_name}")
    
    def detect_and_isolate_product(self, image_path, padding=10):
        """
        Advanced product detection that identifies the main product and removes decorative elements.
        
        Args:
            image_path (str): Path to the image to process
            padding (int): Extra padding around the detected product
        
        Returns:
            PIL.Image: Image with only the main product visible
        """
        # Open the image
        img = Image.open(image_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array for processing
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Get alpha channel
        alpha_channel = img_array[:, :, 3]
        
        try:
            import cv2
            
            # Create binary mask from alpha channel
            binary_mask = (alpha_channel > 0).astype(np.uint8) * 255
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Try to separate bottle from decorative elements by using erosion
            # This helps break connections between the main product and decorative elements
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.erode(binary_mask, erosion_kernel, iterations=1)
            
            # Find components in the eroded mask (should isolate the main bottle better)
            num_labels_eroded, labels_eroded, stats_eroded, centroids_eroded = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
            
            # Use the eroded components if they give us better separation
            if num_labels_eroded > 1:
                logger.info(f"Using eroded mask with {num_labels_eroded} components for better separation")
                num_labels, labels, stats, centroids = num_labels_eroded, labels_eroded, stats_eroded, centroids_eroded
                binary_mask = eroded_mask
            else:
                # Find connected components in original mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            if num_labels <= 1:
                logger.warning("No components found, returning original image")
                return img
            
            # Analyze each component to find the main product
            center_x, center_y = width // 2, height // 2
            best_component = None
            best_score = -1
            
            for i in range(1, num_labels):  # Skip background (label 0)
                # Get component properties
                area = stats[i, cv2.CC_STAT_AREA]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                centroid_x = centroids[i][0]
                centroid_y = centroids[i][1]
                
                # Calculate score based on multiple factors
                # 1. Distance from center (closer is better)
                distance_from_center = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
                center_score = max(0, 1 - (distance_from_center / (width/2)))
                
                # 2. Size (not too small, not too large)
                size_ratio = area / (width * height)
                if size_ratio < 0.005:  # Too small (decorative elements)
                    continue
                if size_ratio > 0.4:  # Too large (probably includes everything)
                    continue
                size_score = 1 - abs(size_ratio - 0.08) / 0.08  # Optimal around 8% of image (more restrictive)
                
                # 3. Aspect ratio (products are usually taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                aspect_score = max(0, 1 - abs(aspect_ratio - 1.5) / 1.5)  # Optimal around 1.5:1
                
                # 4. Position (should be roughly centered)
                position_x_score = max(0, 1 - abs(centroid_x - center_x) / (width/2))
                position_y_score = max(0, 1 - abs(centroid_y - center_y) / (height/2))
                position_score = (position_x_score + position_y_score) / 2
                
                # 5. Compactness (products are usually more compact than scattered elements)
                perimeter = 2 * (w + h)
                compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                compactness_score = compactness
                
                # 6. Shape analysis - products are usually more rectangular/vertical
                # Calculate how well the component fits in its bounding box
                bounding_box_area = w * h
                fill_ratio = area / bounding_box_area if bounding_box_area > 0 else 0
                shape_score = fill_ratio
                
                # 7. Vertical orientation (bottles are usually taller than wide)
                vertical_score = max(0, (h - w) / h) if h > 0 else 0
                
                # Calculate total score
                total_score = (
                    center_score * 0.25 +
                    size_score * 0.2 +
                    aspect_score * 0.15 +
                    position_score * 0.15 +
                    compactness_score * 0.1 +
                    shape_score * 0.1 +
                    vertical_score * 0.05
                )
                
                logger.info(f"Component {i}: area={area}, center_score={center_score:.2f}, size_score={size_score:.2f}, aspect_score={aspect_score:.2f}, total_score={total_score:.2f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_component = i
            
            if best_component is None:
                logger.warning("No suitable component found, using center region")
                # Fallback to center region
                center_region_size = min(width, height) // 4
                left = max(0, center_x - center_region_size)
                top = max(0, center_y - center_region_size)
                right = min(width, center_x + center_region_size)
                bottom = min(height, center_y + center_region_size)
                
                # Create mask for center region
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[top:bottom, left:right] = 255
            else:
                # Create mask for the best component only
                mask = (labels == best_component).astype(np.uint8) * 255
                logger.info(f"Selected component {best_component} with score {best_score:.2f}")
            
            # Apply the mask to the image
            result_array = img_array.copy()
            result_array[:, :, 3] = np.where(mask > 0, img_array[:, :, 3], 0)
            
            # Convert back to PIL Image
            result_img = Image.fromarray(result_array, 'RGBA')
            
            # Auto-crop to remove empty space
            non_transparent = np.where(result_array[:, :, 3] > 0)
            if len(non_transparent[0]) > 0:
                min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
                min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
                
                # Add padding
                min_y = max(0, min_y - padding)
                max_y = min(height, max_y + padding)
                min_x = max(0, min_x - padding)
                max_x = min(width, max_x + padding)
                
                result_img = result_img.crop((min_x, min_y, max_x, max_y))
                logger.info(f"Product isolated and cropped: {min_x},{min_y} to {max_x},{max_y} (size: {max_x-min_x}x{max_y-min_y})")
            
            return result_img
            
        except ImportError:
            logger.warning("OpenCV not available, using fallback method")
            return img
    
    def auto_crop_image(self, image_path, padding=10, focus_on_main_product=True):
        """
        Automatically crop image to remove empty/transparent space around the subject.
        
        Args:
            image_path (str): Path to the image to crop
            padding (int): Extra padding around the subject in pixels
            focus_on_main_product (bool): Whether to focus on the main product (center) only
        
        Returns:
            PIL.Image: Cropped image
        """
        # Open the image
        img = Image.open(image_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array for processing
        img_array = np.array(img)
        
        # Find non-transparent pixels
        alpha_channel = img_array[:, :, 3]
        non_transparent = np.where(alpha_channel > 0)
        
        if len(non_transparent[0]) == 0:
            logger.warning("No non-transparent pixels found, returning original image")
            return img
        
        if focus_on_main_product:
            # Use a more precise method to isolate just the main bottle
            try:
                import cv2
                
                # Create binary mask from alpha channel
                binary_mask = (alpha_channel > 0).astype(np.uint8) * 255
                
                # Apply morphological operations to separate connected objects
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                if num_labels > 1:  # More than just background
                    height, width = img_array.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    
                    # Find the component closest to the center (main product)
                    best_component = 1
                    best_distance = float('inf')
                    
                    for i in range(1, num_labels):
                        centroid_x = centroids[i][0]
                        centroid_y = centroids[i][1]
                        distance = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
                        
                        # Also check if it's reasonably sized (not too small)
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area > (width * height * 0.01) and distance < best_distance:  # At least 1% of image
                            best_distance = distance
                            best_component = i
                    
                    # Get bounding box of best component
                    x = stats[best_component, cv2.CC_STAT_LEFT]
                    y = stats[best_component, cv2.CC_STAT_TOP]
                    w = stats[best_component, cv2.CC_STAT_WIDTH]
                    h = stats[best_component, cv2.CC_STAT_HEIGHT]
                    
                    min_x, min_y = x, y
                    max_x, max_y = x + w, y + h
                    
                    logger.info(f"Found main product component: {w}x{h} at ({x},{y}) - closest to center")
                else:
                    # Fallback to center region method
                    height, width = img_array.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    center_region_size = min(width, height) // 6  # Even smaller region
                    
                    center_mask = (
                        (non_transparent[1] >= center_x - center_region_size) &
                        (non_transparent[1] <= center_x + center_region_size) &
                        (non_transparent[0] >= center_y - center_region_size) &
                        (non_transparent[0] <= center_y + center_region_size)
                    )
                    
                    if np.any(center_mask):
                        center_y_coords = non_transparent[0][center_mask]
                        center_x_coords = non_transparent[1][center_mask]
                        min_y, max_y = center_y_coords.min(), center_y_coords.max()
                        min_x, max_x = center_x_coords.min(), center_x_coords.max()
                        logger.info("Focused on main product in center region")
                    else:
                        min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
                        min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
                        logger.info("Using all non-transparent pixels")
                        
            except ImportError:
                # Fallback if OpenCV not available
                min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
                min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
                logger.info("OpenCV not available, using all non-transparent pixels")
        else:
            # Use all non-transparent pixels
            min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
            min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
        
        # Add padding
        height, width = img_array.shape[:2]
        min_y = max(0, min_y - padding)
        max_y = min(height, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(width, max_x + padding)
        
        # Crop the image
        cropped_img = img.crop((min_x, min_y, max_x, max_y))
        
        logger.info(f"Auto-cropped image: {min_x},{min_y} to {max_x},{max_y} (size: {max_x-min_x}x{max_y-min_y})")
        return cropped_img
    
    def upscale_image(self, image_path, scale_factor=2, method='lanczos', enhance_quality=True):
        """
        Upscale image with high quality enhancement.
        
        Args:
            image_path (str): Path to the image to upscale
            scale_factor (float): Factor to scale the image (2.0 = 2x larger)
            method (str): Upscaling method ('lanczos', 'bicubic', 'nearest', 'cubic')
            enhance_quality (bool): Whether to apply quality enhancement
        
        Returns:
            PIL.Image: Upscaled image
        """
        # Open the image
        img = Image.open(image_path)
        original_size = img.size
        
        # Choose upscaling method
        if method == 'lanczos':
            resample = Image.Resampling.LANCZOS
        elif method == 'bicubic':
            resample = Image.Resampling.BICUBIC
        elif method == 'nearest':
            resample = Image.Resampling.NEAREST
        else:
            resample = Image.Resampling.LANCZOS
        
        # Calculate new size
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
        
        # Upscale the image
        logger.info(f"Upscaling from {original_size[0]}x{original_size[1]} to {new_width}x{new_height}")
        upscaled_img = img.resize((new_width, new_height), resample)
        
        # Apply quality enhancement if requested
        if enhance_quality:
            upscaled_img = self.enhance_image_quality(upscaled_img)
        
        logger.info(f"Upscaled image with {method} method and quality enhancement")
        return upscaled_img
    
    def enhance_image_quality(self, img):
        """
        Enhance image quality using various techniques.
        
        Args:
            img (PIL.Image): Image to enhance
        
        Returns:
            PIL.Image: Enhanced image
        """
        # Convert to RGB if needed (for enhancement operations)
        if img.mode == 'RGBA':
            # Create a white background for enhancement
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            enhanced = background
        else:
            enhanced = img.convert('RGB')
        
        # Apply sharpening filter
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        # Convert back to original mode
        if img.mode == 'RGBA':
            # Convert back to RGBA and restore alpha channel
            enhanced = enhanced.convert('RGBA')
            # Copy the original alpha channel
            enhanced.putalpha(img.split()[-1])
        
        return enhanced
    
    def upscale_with_ai(self, image_path, scale_factor=2):
        """
        Advanced AI-powered upscaling using OpenCV and scikit-image.
        
        Args:
            image_path (str): Path to the image to upscale
            scale_factor (float): Factor to scale the image
        
        Returns:
            PIL.Image: AI-upscaled image
        """
        # Read image with PIL first to preserve RGBA
        original_img = Image.open(image_path)
        original_mode = original_img.mode
        
        # Read image with OpenCV
        img_cv = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        if img_cv is None:
            logger.warning("Could not read image with OpenCV, falling back to PIL upscaling")
            return self.upscale_image(image_path, scale_factor)
        
        original_height, original_width = img_cv.shape[:2]
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        logger.info(f"AI upscaling from {original_width}x{original_height} to {new_width}x{new_height}")
        
        # Use INTER_CUBIC for better quality upscaling
        upscaled_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply denoising only to RGB channels, preserve alpha
        if len(upscaled_cv.shape) == 4:  # RGBA
            # Denoise RGB channels only
            rgb_channels = upscaled_cv[:, :, :3]
            alpha_channel = upscaled_cv[:, :, 3]
            rgb_channels = cv2.fastNlMeansDenoisingColored(rgb_channels, None, 10, 10, 7, 21)
            upscaled_cv = np.dstack([rgb_channels, alpha_channel])
        elif len(upscaled_cv.shape) == 3:  # RGB
            upscaled_cv = cv2.fastNlMeansDenoisingColored(upscaled_cv, None, 10, 10, 7, 21)
        else:  # Grayscale
            upscaled_cv = cv2.fastNlMeansDenoising(upscaled_cv, None, 10, 7, 21)
        
        # Apply sharpening only to RGB channels
        if len(upscaled_cv.shape) == 4:  # RGBA
            rgb_channels = upscaled_cv[:, :, :3]
            alpha_channel = upscaled_cv[:, :, 3]
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            rgb_channels = cv2.filter2D(rgb_channels, -1, kernel)
            upscaled_cv = np.dstack([rgb_channels, alpha_channel])
        elif len(upscaled_cv.shape) == 3:  # RGB
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            upscaled_cv = cv2.filter2D(upscaled_cv, -1, kernel)
        
        # Convert back to PIL Image preserving original mode
        if len(upscaled_cv.shape) == 4:  # RGBA
            upscaled_cv = cv2.cvtColor(upscaled_cv[:, :, :3], cv2.COLOR_BGR2RGB)
            upscaled_img = Image.fromarray(upscaled_cv, 'RGB')
            # Add alpha channel back
            alpha_array = upscaled_cv[:, :, 3] if len(upscaled_cv.shape) == 4 else None
            if alpha_array is not None:
                upscaled_img.putalpha(Image.fromarray(alpha_array, 'L'))
        elif len(upscaled_cv.shape) == 3:  # RGB
            upscaled_cv = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB)
            upscaled_img = Image.fromarray(upscaled_cv, 'RGB')
        else:  # Grayscale
            upscaled_img = Image.fromarray(upscaled_cv, 'L')
        
        # Convert to original mode if needed
        if original_mode == 'RGBA' and upscaled_img.mode != 'RGBA':
            upscaled_img = upscaled_img.convert('RGBA')
        
        logger.info("AI upscaling completed with denoising and sharpening")
        return upscaled_img
    
    def upscale_with_realesrgan(self, image_path, scale_factor=2):
        """
        Ultra-high quality upscaling using Real-ESRGAN AI model.
        
        Args:
            image_path (str): Path to the image to upscale
            scale_factor (float): Factor to scale the image
        
        Returns:
            PIL.Image: Real-ESRGAN upscaled image
        """
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Open the image
            img = Image.open(image_path)
            original_mode = img.mode
            
            # Convert to RGB for Real-ESRGAN
            if img.mode == 'RGBA':
                # Create white background for processing
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img_rgb = background
                alpha_channel = img.split()[-1]
            else:
                img_rgb = img.convert('RGB')
                alpha_channel = None
            
            # Convert to numpy array
            img_array = np.array(img_rgb)
            
            # Initialize Real-ESRGAN
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(scale=4, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model=model, tile=0, tile_pad=10, pre_pad=0, half=True)
            
            # Upscale the image
            logger.info(f"Real-ESRGAN upscaling from {img_array.shape[1]}x{img_array.shape[0]}")
            upscaled_array, _ = upsampler.enhance(img_array, outscale=scale_factor)
            
            # Convert back to PIL Image
            upscaled_img = Image.fromarray(upscaled_array)
            
            # Restore alpha channel if original had one
            if alpha_channel is not None:
                # Resize alpha channel to match upscaled size
                alpha_resized = alpha_channel.resize(upscaled_img.size, Image.Resampling.LANCZOS)
                upscaled_img = upscaled_img.convert('RGBA')
                upscaled_img.putalpha(alpha_resized)
            
            logger.info("Real-ESRGAN upscaling completed")
            return upscaled_img
            
        except ImportError:
            logger.warning("Real-ESRGAN not available, falling back to standard AI upscaling")
            return self.upscale_with_ai(image_path, scale_factor)
        except Exception as e:
            logger.error(f"Real-ESRGAN upscaling failed: {str(e)}, falling back to standard AI upscaling")
            return self.upscale_with_ai(image_path, scale_factor)
    
    def upscale_with_gfpgan(self, image_path, scale_factor=2):
        """
        Ultra-high quality upscaling using GFPGAN for face and general image enhancement.
        
        Args:
            image_path (str): Path to the image to upscale
            scale_factor (float): Factor to scale the image
        
        Returns:
            PIL.Image: GFPGAN upscaled image
        """
        try:
            from gfpgan import GFPGANer
            
            # Open the image
            img = Image.open(image_path)
            original_mode = img.mode
            
            # Convert to RGB for GFPGAN
            if img.mode == 'RGBA':
                # Create white background for processing
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img_rgb = background
                alpha_channel = img.split()[-1]
            else:
                img_rgb = img.convert('RGB')
                alpha_channel = None
            
            # Convert to numpy array
            img_array = np.array(img_rgb)
            
            # Initialize GFPGAN
            restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=scale_factor, arch='clean', channel_multiplier=2, bg_upsampler=None)
            
            # Upscale the image
            logger.info(f"GFPGAN upscaling from {img_array.shape[1]}x{img_array.shape[0]}")
            _, _, upscaled_array = restorer.enhance(img_array, has_aligned=False, only_center_face=False, paste_back=True)
            
            # Convert back to PIL Image
            upscaled_img = Image.fromarray(upscaled_array)
            
            # Restore alpha channel if original had one
            if alpha_channel is not None:
                # Resize alpha channel to match upscaled size
                alpha_resized = alpha_channel.resize(upscaled_img.size, Image.Resampling.LANCZOS)
                upscaled_img = upscaled_img.convert('RGBA')
                upscaled_img.putalpha(alpha_resized)
            
            logger.info("GFPGAN upscaling completed")
            return upscaled_img
            
        except ImportError:
            logger.warning("GFPGAN not available, falling back to standard AI upscaling")
            return self.upscale_with_ai(image_path, scale_factor)
        except Exception as e:
            logger.error(f"GFPGAN upscaling failed: {str(e)}, falling back to standard AI upscaling")
            return self.upscale_with_ai(image_path, scale_factor)
    
    def remove_background(self, input_path, output_path=None, format='PNG', auto_crop=True, padding=10, upscale=False, scale_factor=2, upscale_method='lanczos'):
        """
        Remove background from a single image.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            format (str): Output format ('PNG' for transparency, 'JPEG' for white background)
            auto_crop (bool): Whether to automatically crop empty space
            padding (int): Padding around the subject when auto-cropping
            upscale (bool): Whether to upscale the image
            scale_factor (float): Factor to scale the image
            upscale_method (str): Upscaling method ('lanczos', 'bicubic', 'ai')
        
        Returns:
            str: Path to the output image
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_no_bg.{format.lower()}"
        else:
            output_path = Path(output_path)
        
        try:
            # Read input image
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
            
            # Remove background
            logger.info(f"Processing: {input_path.name}")
            output_data = remove(input_data, session=self.session)
            
            # Save initial output
            temp_output = output_path.parent / f"temp_{output_path.name}"
            with open(temp_output, 'wb') as output_file:
                output_file.write(output_data)
            
            # Auto-crop if requested
            if auto_crop:
                logger.info("Detecting and isolating main product")
                # Use advanced product detection to isolate only the main product
                cropped_img = self.detect_and_isolate_product(temp_output, padding)
                cropped_img.save(temp_output, format)
            
            # Upscale if requested
            if upscale:
                logger.info(f"Upscaling image by factor {scale_factor} using {upscale_method} method")
                if upscale_method == 'realesrgan':
                    upscaled_img = self.upscale_with_realesrgan(temp_output, scale_factor)
                elif upscale_method == 'gfpgan':
                    upscaled_img = self.upscale_with_gfpgan(temp_output, scale_factor)
                elif upscale_method == 'ai':
                    upscaled_img = self.upscale_with_ai(temp_output, scale_factor)
                else:
                    upscaled_img = self.upscale_image(temp_output, scale_factor, upscale_method)
                upscaled_img.save(output_path, format)
                temp_output.unlink()  # Remove temp file
            else:
                # Just rename temp file to final output
                temp_output.rename(output_path)
            
            logger.info(f"Saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise
    
    def remove_background_with_white_bg(self, input_path, output_path=None, auto_crop=True, padding=10, upscale=False, scale_factor=2, upscale_method='lanczos'):
        """
        Remove background and replace with white background.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            auto_crop (bool): Whether to automatically crop empty space
            padding (int): Padding around the subject when auto-cropping
            upscale (bool): Whether to upscale the image
            scale_factor (float): Factor to scale the image
            upscale_method (str): Upscaling method ('lanczos', 'bicubic', 'ai')
        
        Returns:
            str: Path to the output image
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_white_bg.png"
        else:
            output_path = Path(output_path)
        
        try:
            # First remove background (creates transparent PNG)
            temp_path = input_path.parent / f"{input_path.stem}_temp.png"
            self.remove_background(input_path, temp_path, auto_crop=False)  # Don't crop yet
            
            # Open the transparent image
            transparent_img = Image.open(temp_path)
            
            # Auto-crop if requested
            if auto_crop:
                logger.info("Detecting and isolating main product")
                # Use advanced product detection to isolate only the main product
                transparent_img = self.detect_and_isolate_product(temp_path, padding)
            
            # Upscale if requested
            if upscale:
                logger.info(f"Upscaling image by factor {scale_factor} using {upscale_method} method")
                # Save the cropped image temporarily for upscaling
                temp_cropped = temp_path.parent / f"temp_cropped_{temp_path.name}"
                transparent_img.save(temp_cropped)
                if upscale_method == 'realesrgan':
                    transparent_img = self.upscale_with_realesrgan(temp_cropped, scale_factor)
                elif upscale_method == 'gfpgan':
                    transparent_img = self.upscale_with_gfpgan(temp_cropped, scale_factor)
                elif upscale_method == 'ai':
                    transparent_img = self.upscale_with_ai(temp_cropped, scale_factor)
                else:
                    transparent_img = self.upscale_image(temp_cropped, scale_factor, upscale_method)
                temp_cropped.unlink()  # Clean up temp file
            
            # Create white background
            white_bg = Image.new('RGB', transparent_img.size, (255, 255, 255))
            
            # Paste transparent image on white background
            if transparent_img.mode == 'RGBA':
                white_bg.paste(transparent_img, mask=transparent_img.split()[-1])
            else:
                white_bg.paste(transparent_img)
            
            # Save final image
            white_bg.save(output_path, 'PNG')
            
            # Clean up temp file
            temp_path.unlink()
            
            logger.info(f"Saved with white background: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise
    
    def batch_remove_background(self, input_dir, output_dir=None, format='PNG', white_bg=False, auto_crop=True, padding=10, upscale=False, scale_factor=2, upscale_method='lanczos'):
        """
        Remove background from all images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save output images (optional)
            format (str): Output format
            white_bg (bool): Whether to use white background instead of transparency
            auto_crop (bool): Whether to automatically crop empty space
            padding (int): Padding around the subject when auto-cropping
            upscale (bool): Whether to upscale the image
            scale_factor (float): Factor to scale the image
            upscale_method (str): Upscaling method ('lanczos', 'bicubic', 'ai')
        
        Returns:
            list: List of output file paths
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Set output directory
        if output_dir is None:
            output_dir = input_dir / "no_background"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = [f for f in input_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        output_paths = []
        for image_file in image_files:
            try:
                if white_bg:
                    output_path = self.remove_background_with_white_bg(
                        image_file, 
                        output_dir / f"{image_file.stem}_white_bg.png",
                        auto_crop=auto_crop,
                        padding=padding,
                        upscale=upscale,
                        scale_factor=scale_factor,
                        upscale_method=upscale_method
                    )
                else:
                    output_path = self.remove_background(
                        image_file, 
                        output_dir / f"{image_file.stem}_no_bg.{format.lower()}",
                        auto_crop=auto_crop,
                        padding=padding,
                        upscale=upscale,
                        scale_factor=scale_factor,
                        upscale_method=upscale_method
                    )
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {str(e)}")
        
        logger.info(f"Successfully processed {len(output_paths)} images")
        return output_paths

def main():
    parser = argparse.ArgumentParser(description='Remove backgrounds from images')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-m', '--model', default='u2net', 
                       choices=['u2net', 'u2net_human_seg', 'u2netp', 'silueta', 'isnet-general-use'],
                       help='Model to use for background removal')
    parser.add_argument('-f', '--format', default='PNG', choices=['PNG', 'JPEG'],
                       help='Output format (PNG for transparency, JPEG for white background)')
    parser.add_argument('--white-bg', action='store_true',
                       help='Replace transparent background with white background')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    parser.add_argument('--no-crop', action='store_true',
                       help='Disable automatic cropping (keep original image size)')
    parser.add_argument('--padding', type=int, default=10,
                       help='Padding around subject when auto-cropping (default: 10 pixels)')
    parser.add_argument('--upscale', action='store_true',
                       help='Upscale the image for higher quality')
    parser.add_argument('--scale-factor', type=float, default=2.0,
                       help='Scale factor for upscaling (default: 2.0 = 2x larger)')
    parser.add_argument('--upscale-method', default='lanczos',
                       choices=['lanczos', 'bicubic', 'nearest', 'ai', 'realesrgan', 'gfpgan'],
                       help='Upscaling method (default: lanczos). realesrgan and gfpgan are ultra-high quality AI methods')
    
    args = parser.parse_args()
    
    try:
        # Initialize background remover
        remover = BackgroundRemover(model_name=args.model)
        
        # Set processing parameters
        auto_crop = not args.no_crop
        padding = args.padding
        upscale = args.upscale
        scale_factor = args.scale_factor
        upscale_method = args.upscale_method
        
        input_path = Path(args.input)
        
        if args.batch or input_path.is_dir():
            # Batch processing
            output_paths = remover.batch_remove_background(
                args.input, 
                args.output, 
                args.format, 
                args.white_bg,
                auto_crop=auto_crop,
                padding=padding,
                upscale=upscale,
                scale_factor=scale_factor,
                upscale_method=upscale_method
            )
            print(f"Processed {len(output_paths)} images")
        else:
            # Single file processing
            if args.white_bg:
                output_path = remover.remove_background_with_white_bg(
                    args.input, 
                    args.output,
                    auto_crop=auto_crop,
                    padding=padding,
                    upscale=upscale,
                    scale_factor=scale_factor,
                    upscale_method=upscale_method
                )
            else:
                output_path = remover.remove_background(
                    args.input, 
                    args.output, 
                    args.format,
                    auto_crop=auto_crop,
                    padding=padding,
                    upscale=upscale,
                    scale_factor=scale_factor,
                    upscale_method=upscale_method
                )
            print(f"Processed: {output_path}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
