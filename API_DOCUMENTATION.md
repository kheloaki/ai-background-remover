# AI Background Remover API Documentation

## 🚀 Base URL
```
https://your-railway-app.railway.app
```

## 📋 Available Endpoints

### 1. Health Check
**GET** `/health`
- **Description**: Check if the service is running
- **Response**: 
```json
{
  "status": "healthy",
  "service": "background-remover"
}
```

### 2. Get Available Models
**GET** `/api/models`
- **Description**: Get list of available AI models
- **Response**:
```json
{
  "models": [
    {
      "id": "isnet-general-use",
      "name": "ISNet General Use",
      "description": "Best for general purpose background removal"
    },
    {
      "id": "u2net",
      "name": "U²-Net",
      "description": "Good for most images"
    }
  ]
}
```

### 3. Get Upscale Methods
**GET** `/api/upscale-methods`
- **Description**: Get list of available upscaling methods
- **Response**:
```json
{
  "methods": [
    {
      "id": "lanczos",
      "name": "Lanczos",
      "description": "Fast, good quality, preserves transparency"
    },
    {
      "id": "ai",
      "name": "AI Enhanced",
      "description": "Advanced AI upscaling with denoising and sharpening"
    }
  ]
}
```

### 4. Remove Background (File Upload)
**POST** `/api/remove-background`
- **Description**: Remove background from uploaded image file
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (file, required): Image file to process
  - `model` (string, optional): AI model to use (default: "isnet-general-use")
  - `upscale` (boolean, optional): Enable upscaling (default: false)
  - `scale_factor` (float, optional): Scale factor for upscaling (default: 2.0)
  - `upscale_method` (string, optional): Upscaling method (default: "lanczos")
  - `padding` (integer, optional): Padding around product (default: 10)
  - `white_bg` (boolean, optional): Use white background instead of transparent (default: false)

- **Response**: Returns processed image file

### 5. Remove Background (Base64)
**POST** `/api/remove-background-base64`
- **Description**: Remove background from base64 encoded image
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "model": "isnet-general-use",
  "upscale": true,
  "scale_factor": 2.0,
  "upscale_method": "lanczos",
  "padding": 10,
  "white_bg": false
}
```

- **Response**:
```json
{
  "success": true,
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "message": "Background removed successfully"
}
```

### 6. Batch Remove Background
**POST** `/api/batch-remove`
- **Description**: Process multiple images at once
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `files` (file[], required): Multiple image files
  - `model` (string, optional): AI model to use
  - `upscale` (boolean, optional): Enable upscaling
  - `scale_factor` (float, optional): Scale factor for upscaling
  - `upscale_method` (string, optional): Upscaling method
  - `padding` (integer, optional): Padding around product
  - `white_bg` (boolean, optional): Use white background

- **Response**: Returns ZIP file with all processed images

## 🔧 Usage Examples

### Python Example (Base64)
```python
import requests
import base64

# Read image file
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# API request
response = requests.post('https://your-app.railway.app/api/remove-background-base64', 
    json={
        'image': f'data:image/jpeg;base64,{image_data}',
        'model': 'isnet-general-use',
        'upscale': True,
        'scale_factor': 2.0,
        'upscale_method': 'lanczos',
        'padding': 10,
        'white_bg': False
    }
)

if response.status_code == 200:
    result = response.json()
    # Save result image
    result_data = result['image'].split(',')[1]
    with open('result.png', 'wb') as f:
        f.write(base64.b64decode(result_data))
```

### Python Example (File Upload)
```python
import requests

# File upload
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'model': 'isnet-general-use',
        'upscale': 'true',
        'scale_factor': '2.0',
        'upscale_method': 'lanczos',
        'padding': '10',
        'white_bg': 'false'
    }
    
    response = requests.post('https://your-app.railway.app/api/remove-background', 
                           files=files, data=data)

if response.status_code == 200:
    with open('result.png', 'wb') as f:
        f.write(response.content)
```

### JavaScript Example
```javascript
// Base64 API call
async function removeBackground(imageFile) {
    const reader = new FileReader();
    reader.onload = async function(e) {
        const response = await fetch('https://your-app.railway.app/api/remove-background-base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: e.target.result,
                model: 'isnet-general-use',
                upscale: true,
                scale_factor: 2.0,
                upscale_method: 'lanczos',
                padding: 10,
                white_bg: false
            })
        });
        
        const result = await response.json();
        if (result.success) {
            // Display result image
            const img = document.createElement('img');
            img.src = result.image;
            document.body.appendChild(img);
        }
    };
    reader.readAsDataURL(imageFile);
}
```

### cURL Examples
```bash
# File upload
curl -X POST https://your-app.railway.app/api/remove-background \
  -F "file=@image.jpg" \
  -F "model=isnet-general-use" \
  -F "upscale=true" \
  -F "scale_factor=2.0" \
  -F "upscale_method=lanczos" \
  -F "padding=10" \
  -F "white_bg=false" \
  --output result.png

# Base64
curl -X POST https://your-app.railway.app/api/remove-background-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "model": "isnet-general-use",
    "upscale": true,
    "scale_factor": 2.0,
    "upscale_method": "lanczos",
    "padding": 10,
    "white_bg": false
  }'
```

## 🎯 AI Models Available

| Model ID | Name | Best For |
|----------|------|----------|
| `isnet-general-use` | ISNet General Use | General purpose (recommended) |
| `u2net` | U²-Net | Most images |
| `u2netp` | U²-Net+ | Lightweight processing |
| `u2net_human_seg` | U²-Net Human Segmentation | Human subjects |
| `u2net_cloth_seg` | U²-Net Cloth Segmentation | Clothing items |
| `silueta` | Silueta | Portraits and people |
| `isnet-anime` | ISNet Anime | Anime and cartoon images |

## ⚡ Upscaling Methods

| Method | Description | Speed | Quality | Transparency |
|--------|-------------|-------|---------|--------------|
| `lanczos` | Fast Lanczos interpolation | Fast | Good | ✅ Preserved |
| `bicubic` | Bicubic interpolation | Medium | Good | ✅ Preserved |
| `ai` | AI-enhanced upscaling | Slow | Excellent | ❌ May lose |

## 📊 Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 500 | Internal Server Error |

## 🔒 Rate Limits
- No rate limits currently implemented
- Processing time varies by image size and model complexity
- Large images may take longer to process

## 📝 Notes
- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG, WEBP, GIF, BMP, TIFF
- Output format: PNG (with transparency unless white_bg=true)
- Auto-cropping is enabled by default to remove empty space
- Product detection automatically isolates the main subject
