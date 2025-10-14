#!/usr/bin/env python3
"""
Lightweight Background Remover for Railway Deployment
Optimized for smaller Docker images
"""

import os
import logging
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from rembg import remove, new_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightBackgroundRemover:
    """Lightweight background remover optimized for deployment"""
    
    def __init__(self, model_name='u2net'):
        """
        Initialize the background remover with a lightweight model
        
        Args:
            model_name (str): The model to use for background removal.
                             Options: 'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta'
        """
        self.model_name = model_name
        try:
            self.session = new_session(model_name)
            logger.info(f"Initialized lightweight background remover with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            # Fallback to u2netp (lightest model)
            try:
                self.model_name = 'u2netp'
                self.session = new_session('u2netp')
                logger.info(f"Fallback to model: u2netp")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback model: {e2}")
                self.session = None
    
    def remove_background(self, input_path, output_path=None, auto_crop=True, padding=10):
        """
        Remove background from an image using lightweight processing
        
        Args:
            input_path (str): Path to input image
            output_path (str, optional): Path for output image
            auto_crop (bool): Whether to auto-crop the result
            padding (int): Padding around the cropped image
            
        Returns:
            str: Path to the output image
        """
        if not self.session:
            raise Exception("Background remover not properly initialized")
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_no_bg.png"
        else:
            output_path = Path(output_path)
        
        try:
            # Read input image
            with open(input_path, 'rb') as f:
                input_data = f.read()
            
            # Remove background
            output_data = remove(input_data, session=self.session)
            
            # Save result
            with open(output_path, 'wb') as f:
                f.write(output_data)
            
            # Auto-crop if requested
            if auto_crop:
                self._auto_crop_image(output_path, padding)
            
            logger.info(f"Background removed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise
    
    def remove_background_with_white_bg(self, input_path, output_path=None, auto_crop=True, padding=10):
        """
        Remove background and add white background
        
        Args:
            input_path (str): Path to input image
            output_path (str, optional): Path for output image
            auto_crop (bool): Whether to auto-crop the result
            padding (int): Padding around the cropped image
            
        Returns:
            str: Path to the output image
        """
        # First remove background
        temp_output = self.remove_background(input_path, auto_crop=auto_crop, padding=padding)
        
        # Add white background
        with Image.open(temp_output) as img:
            if img.mode == 'RGBA':
                # Create white background
                white_bg = Image.new('RGB', img.size, (255, 255, 255))
                white_bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                white_bg.save(output_path or temp_output, 'PNG')
        
        return output_path or temp_output
    
    def _auto_crop_image(self, image_path, padding=10):
        """
        Auto-crop image to remove empty space around the subject
        
        Args:
            image_path (str): Path to the image file
            padding (int): Padding to add around the cropped area
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGBA':
                    return
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Find non-transparent pixels
                alpha = img_array[:, :, 3]
                non_transparent = np.where(alpha > 0)
                
                if len(non_transparent[0]) == 0:
                    return  # No non-transparent pixels
                
                # Get bounding box
                min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
                min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
                
                # Add padding
                min_y = max(0, min_y - padding)
                max_y = min(img.height, max_y + padding)
                min_x = max(0, min_x - padding)
                max_x = min(img.width, max_x + padding)
                
                # Crop the image
                cropped = img.crop((min_x, min_y, max_x, max_y))
                cropped.save(image_path)
                
                logger.info(f"Auto-cropped image: {min_x},{min_y} to {max_x},{max_y}")
                
        except Exception as e:
            logger.error(f"Error auto-cropping image: {e}")
    
    def enhance_image(self, image_path, output_path=None):
        """
        Enhance image quality with lightweight processing
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path for output image
            
        Returns:
            str: Path to the enhanced image
        """
        if output_path is None:
            output_path = image_path
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode == 'RGBA':
                    # Create white background for enhancement
                    white_bg = Image.new('RGB', img.size, (255, 255, 255))
                    white_bg.paste(img, mask=img.split()[-1])
                    img = white_bg
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                # Save enhanced image
                img.save(output_path, 'PNG')
                
                logger.info(f"Image enhanced: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path
    
    def batch_remove_background(self, input_dir, output_dir=None, auto_crop=True, padding=10):
        """
        Remove background from multiple images
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str, optional): Directory for output images
            auto_crop (bool): Whether to auto-crop results
            padding (int): Padding around cropped images
            
        Returns:
            list: List of processed image paths
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_path / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        
        processed_files = []
        
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                try:
                    output_path = output_dir / f"{file_path.stem}_no_bg.png"
                    self.remove_background(file_path, output_path, auto_crop, padding)
                    processed_files.append(str(output_path))
                    logger.info(f"Processed: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        logger.info(f"Batch processing completed. {len(processed_files)} files processed.")
        return processed_files

def main():
    """Command line interface for the lightweight background remover"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lightweight AI Background Remover')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-m', '--model', default='u2net', 
                       choices=['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta'],
                       help='AI model to use')
    parser.add_argument('--no-crop', action='store_true', help='Disable auto-cropping')
    parser.add_argument('-p', '--padding', type=int, default=10, help='Padding around cropped image')
    parser.add_argument('--white-bg', action='store_true', help='Add white background')
    parser.add_argument('--enhance', action='store_true', help='Enhance image quality')
    
    args = parser.parse_args()
    
    # Initialize remover
    remover = LightweightBackgroundRemover(args.model)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        try:
            if args.white_bg:
                result = remover.remove_background_with_white_bg(
                    input_path, args.output, not args.no_crop, args.padding
                )
            else:
                result = remover.remove_background(
                    input_path, args.output, not args.no_crop, args.padding
                )
            
            if args.enhance:
                remover.enhance_image(result)
            
            print(f"✅ Processed: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif input_path.is_dir():
        # Batch processing
        try:
            results = remover.batch_remove_background(
                input_path, args.output, not args.no_crop, args.padding
            )
            print(f"✅ Batch processed {len(results)} files")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print(f"❌ Input path not found: {input_path}")

if __name__ == "__main__":
    main()
