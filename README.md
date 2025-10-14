# Background Removal Script

A Python script that automatically removes backgrounds from images using AI-powered background removal technology. Perfect for product photography, e-commerce, and any images that need clean, professional backgrounds.

## Features

- **AI-Powered Background Removal**: Uses state-of-the-art deep learning models
- **Multiple Models**: Choose from different models optimized for different use cases
- **Batch Processing**: Process entire folders of images at once
- **Flexible Output**: Save with transparent backgrounds or white backgrounds
- **Multiple Formats**: Support for PNG, JPEG, and other common image formats
- **Command Line Interface**: Easy to use from terminal or integrate into workflows

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python background_remover.py --help
   ```

## Usage

### Basic Usage

**Remove background from a single image**:
```bash
python background_remover.py input_image.jpg
```

**Specify output file**:
```bash
python background_remover.py input_image.jpg -o output_image.png
```

**Remove background and add white background**:
```bash
python background_remover.py input_image.jpg --white-bg
```

### Batch Processing

**Process all images in a folder**:
```bash
python background_remover.py /path/to/images/ --batch
```

**Batch process with white backgrounds**:
```bash
python background_remover.py /path/to/images/ --batch --white-bg
```

### Advanced Options

**Choose different AI model**:
```bash
python background_remover.py input_image.jpg -m u2net_human_seg
```

**Available models**:
- `u2net` (default) - General purpose, good for most images
- `u2net_human_seg` - Optimized for people/portraits
- `u2netp` - Lighter version of u2net
- `silueta` - Good for silhouettes
- `isnet-general-use` - High quality general purpose

**Specify output format**:
```bash
python background_remover.py input_image.jpg -f PNG
```

## Examples

### Product Photography
Perfect for e-commerce product images like the vial you uploaded:

```bash
# Remove background with transparency
python background_remover.py product_vial.jpg

# Remove background with white background
python background_remover.py product_vial.jpg --white-bg
```

### Batch Processing Product Images
```bash
# Process entire product folder
python background_remover.py ./product_images/ --batch --white-bg
```

### Portrait Photography
```bash
# Use human-optimized model for portraits
python background_remover.py portrait.jpg -m u2net_human_seg
```

## Output

- **Transparent Background**: Default output saves as PNG with transparent background
- **White Background**: Use `--white-bg` flag to replace transparency with white
- **Batch Output**: Creates a `no_background` folder with processed images

## Supported Input Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Performance Tips

1. **Model Selection**: 
   - Use `u2net` for general images
   - Use `u2net_human_seg` for people/portraits
   - Use `u2netp` for faster processing

2. **Batch Processing**: More efficient for multiple images

3. **Image Size**: Larger images take longer to process

## Troubleshooting

**Common Issues**:

1. **"No module named 'rembg'"**: Run `pip install -r requirements.txt`

2. **CUDA errors**: The script will automatically fall back to CPU processing

3. **Memory issues**: Try using the lighter `u2netp` model

4. **Slow processing**: First run downloads the AI model (one-time setup)

## Integration

You can also use this script programmatically:

```python
from background_remover import BackgroundRemover

# Initialize
remover = BackgroundRemover()

# Remove background
output_path = remover.remove_background('input.jpg')

# Remove background with white background
output_path = remover.remove_background_with_white_bg('input.jpg')

# Batch process
output_paths = remover.batch_remove_background('./images/', white_bg=True)
```

## License

This script uses the `rembg` library which is licensed under the MIT License.
