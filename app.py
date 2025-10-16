from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import base64
from io import BytesIO
from background_remover_light import LightweightBackgroundRemover
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend/out', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})

# Initialize background remover
bg_remover = None

try:
    bg_remover = LightweightBackgroundRemover()
    logger.info("Lightweight background remover initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize background remover: {e}")

# Serve Next.js frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "background-remover"
    })

@app.route('/api/models')
def get_models():
    """API endpoint to get available models"""
    models = [
        {
            "id": "isnet-general-use",
            "name": "ISNet General Use",
            "description": "Best for general purpose background removal"
        },
        {
            "id": "u2net",
            "name": "U²-Net",
            "description": "Good for most images"
        },
        {
            "id": "u2netp",
            "name": "U²-Net+",
            "description": "Faster but less accurate"
        },
        {
            "id": "u2net_human_seg",
            "name": "U²-Net Human Segmentation",
            "description": "Optimized for human subjects"
        },
        {
            "id": "u2net_cloth_seg",
            "name": "U²-Net Cloth Segmentation",
            "description": "Optimized for clothing"
        },
        {
            "id": "silueta",
            "name": "Silueta",
            "description": "Good for silhouettes"
        },
        {
            "id": "isnet-anime",
            "name": "ISNet Anime",
            "description": "Optimized for anime/cartoon images"
        }
    ]
    return jsonify({"models": models})

@app.route('/api/features')
def get_features():
    """API endpoint to get available features"""
    features = {
        "upscaling": True,
        "white_background": True,
        "auto_crop": True,
        "response_types": ["base64", "url"]
    }
    return jsonify({"features": features})

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    """API endpoint for background removal with file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        model = request.form.get('model', 'u2net')
        padding = int(request.form.get('padding', 10))
        white_bg = request.form.get('white_bg', 'false').lower() == 'true'
        enhance = request.form.get('enhance', 'false').lower() == 'true'
        response_type = request.form.get('response_type', 'base64')
        
        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        input_path = f"temp_{filename}"
        file.save(input_path)
        
        try:
            # Process the image
            if white_bg:
                result_path = bg_remover.remove_background_with_white_bg(
                    input_path,
                    auto_crop=True,
                    padding=padding
                )
            else:
                result_path = bg_remover.remove_background(
                    input_path,
                    auto_crop=True,
                    padding=padding
                )
            
            # Enhance if requested
            if enhance:
                bg_remover.enhance_image(result_path)
            
            if response_type == 'url':
                # Move to files directory
                output_filename = f"processed_{filename}"
                output_path = os.path.join('files', output_filename)
                os.makedirs('files', exist_ok=True)
                os.rename(result_path, output_path)
                
                return jsonify({
                    "success": True,
                    "image_url": f"/files/{output_filename}",
                    "message": "Background removed successfully"
                })
            else:
                # Return base64
                with open(result_path, 'rb') as f:
                    result_bytes = f.read()
                    result_base64 = base64.b64encode(result_bytes).decode('utf-8')
                
                return jsonify({
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "message": "Background removed successfully"
                })
        
        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.remove(input_path)
            if response_type != 'url' and os.path.exists(result_path):
                os.remove(result_path)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/remove-background-base64', methods=['POST'])
def remove_background_base64():
    """API endpoint for background removal with base64 input"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        model = data.get('model', 'u2net')
        padding = int(data.get('padding', 10))
        white_bg = data.get('white_bg', False)
        enhance = data.get('enhance', False)
        response_type = data.get('response_type', 'base64')
        
        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500
        
        # Decode base64
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporarily
        filename = "temp_base64.png"
        input_path = f"temp_{filename}"
        image.save(input_path)
        
        try:
            # Process
            if white_bg:
                result_path = bg_remover.remove_background_with_white_bg(
                    input_path,
                    auto_crop=True,
                    padding=padding
                )
            else:
                result_path = bg_remover.remove_background(
                    input_path,
                    auto_crop=True,
                    padding=padding
                )
            
            if enhance:
                bg_remover.enhance_image(result_path)
            
            if response_type == 'url':
                output_filename = f"processed_{filename}"
                output_path = os.path.join('files', output_filename)
                os.makedirs('files', exist_ok=True)
                os.rename(result_path, output_path)
                
                return jsonify({
                    "success": True,
                    "image_url": f"/files/{output_filename}",
                    "message": "Background removed successfully"
                })
            else:
                with open(result_path, 'rb') as f:
                    result_bytes = f.read()
                    result_base64 = base64.b64encode(result_bytes).decode('utf-8')
                
                return jsonify({
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "message": "Background removed successfully"
                })
        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if response_type != 'url' and os.path.exists(result_path):
                os.remove(result_path)
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/remove-background-url', methods=['POST'])
def remove_background_url():
    """API endpoint for background removal with image URL input"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "No image URL provided"}), 400

        image_url = data['url']
        model = data.get('model', 'u2net')
        padding = int(data.get('padding', 10))
        white_bg = data.get('white_bg', False)
        enhance = data.get('enhance', False)
        response_type = data.get('response_type', 'base64')

        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500

        # Download image
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            filename = secure_filename(image_url.split('/')[-1] or "downloaded_image.png")
            input_path = f"temp_{filename}"
            image.save(input_path)

            try:
                if white_bg:
                    result_path = bg_remover.remove_background_with_white_bg(
                        input_path,
                        auto_crop=True,
                        padding=padding
                    )
                else:
                    result_path = bg_remover.remove_background(
                        input_path,
                        auto_crop=True,
                        padding=padding
                    )
                
                if enhance:
                    bg_remover.enhance_image(result_path)

                if response_type == 'url':
                    output_filename = f"processed_{filename}"
                    output_path = os.path.join('files', output_filename)
                    os.makedirs('files', exist_ok=True)
                    os.rename(result_path, output_path)
                    return jsonify({
                        "success": True,
                        "image_url": f"/files/{output_filename}",
                        "message": "Background removed successfully"
                    })
                else:
                    with open(result_path, 'rb') as f:
                        result_bytes = f.read()
                        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
                    return jsonify({
                        "success": True,
                        "image": f"data:image/png;base64,{result_base64}",
                        "message": "Background removed successfully"
                    })

            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if response_type != 'url' and os.path.exists(result_path):
                    os.remove(result_path)

        except requests.exceptions.RequestException as req_e:
            return jsonify({"error": f"Failed to download image: {str(req_e)}"}), 400
        except Exception as img_e:
            return jsonify({"error": f"Invalid image: {str(img_e)}"}), 400

    except Exception as e:
        logger.error(f"Error processing URL image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/files/<filename>')
def serve_file(filename):
    """Serves processed files"""
    return send_from_directory('files', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)