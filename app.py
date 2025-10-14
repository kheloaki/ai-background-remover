#!/usr/bin/env python3
"""
Background Removal Web Service
Deployable on Railway with web interface
"""

import os
import io
import base64
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import logging
from background_remover_light import LightweightBackgroundRemover

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize lightweight background remover
try:
    bg_remover = LightweightBackgroundRemover()
    logger.info("Lightweight background remover initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize background remover: {e}")
    bg_remover = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "background-remover"})

@app.route('/api/remove-background', methods=['POST'])
def remove_background_api():
    """API endpoint for background removal"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500
        
        # Get parameters from request
        model = request.form.get('model', 'u2net')
        padding = int(request.form.get('padding', '10'))
        white_bg = request.form.get('white_bg', 'false').lower() == 'true'
        enhance = request.form.get('enhance', 'false').lower() == 'true'
        
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
            
            # Enhance image if requested
            if enhance:
                bg_remover.enhance_image(result_path)
            
            # Return the processed image
            return send_file(result_path, as_attachment=True, download_name=f"processed_{filename}")
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(result_path):
                os.remove(result_path)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/remove-background-base64', methods=['POST'])
def remove_background_base64():
    """API endpoint for background removal with base64 input/output"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get parameters from request
        model = data.get('model', 'u2net')
        padding = int(data.get('padding', 10))
        white_bg = data.get('white_bg', False)
        enhance = data.get('enhance', False)
        
        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500
        
        # Decode base64 image
        import base64
        from io import BytesIO
        from PIL import Image
        
        try:
            # Remove data URL prefix if present
            image_data = data['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Save temporarily
            input_path = "temp_input.png"
            image.save(input_path)
            
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
                
                # Enhance image if requested
                if enhance:
                    bg_remover.enhance_image(result_path)
                
                # Convert result to base64
                with open(result_path, 'rb') as f:
                    result_bytes = f.read()
                    result_base64 = base64.b64encode(result_bytes).decode('utf-8')
                
                return jsonify({
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "message": "Background removed successfully"
                })
                
            finally:
                # Clean up temporary files
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(result_path):
                    os.remove(result_path)
        
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available AI models"""
    models = [
        {"id": "u2net", "name": "U²-Net", "description": "Good for most images (recommended)"},
        {"id": "u2netp", "name": "U²-Net+", "description": "Lightweight version of U²-Net (fastest)"},
        {"id": "u2net_human_seg", "name": "U²-Net Human Segmentation", "description": "Optimized for human subjects"},
        {"id": "u2net_cloth_seg", "name": "U²-Net Cloth Segmentation", "description": "Optimized for clothing"},
        {"id": "silueta", "name": "Silueta", "description": "Good for portraits and people"}
    ]
    return jsonify({"models": models})

@app.route('/api/features', methods=['GET'])
def get_available_features():
    """Get list of available features"""
    features = [
        {"id": "auto_crop", "name": "Auto Crop", "description": "Automatically crop to remove empty space"},
        {"id": "white_bg", "name": "White Background", "description": "Add white background instead of transparent"},
        {"id": "enhance", "name": "Image Enhancement", "description": "Enhance image quality with contrast and sharpness"},
        {"id": "padding", "name": "Padding", "description": "Add padding around the cropped subject"}
    ]
    return jsonify({"features": features})

@app.route('/api/batch-remove', methods=['POST'])
def batch_remove_background():
    """API endpoint for batch background removal"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        if not bg_remover:
            return jsonify({"error": "Background remover not available"}), 500
        
        # Get parameters
        model = request.form.get('model', 'u2net')
        padding = int(request.form.get('padding', '10'))
        white_bg = request.form.get('white_bg', 'false').lower() == 'true'
        enhance = request.form.get('enhance', 'false').lower() == 'true'
        
        # Process all files
        results = []
        temp_files = []
        
        try:
            for file in files:
                if file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    input_path = f"temp_{filename}"
                    file.save(input_path)
                    temp_files.append(input_path)
                    
                    # Process image
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
                    
                    # Enhance image if requested
                    if enhance:
                        bg_remover.enhance_image(result_path)
                    
                    results.append({
                        'original': filename,
                        'processed': result_path
                    })
            
            # Create zip file with all results
            import zipfile
            zip_path = "batch_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for result in results:
                    zipf.write(result['processed'], f"processed_{result['original']}")
            
            return send_file(zip_path, as_attachment=True, download_name="batch_results.zip")
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for result in results:
                if os.path.exists(result['processed']):
                    os.remove(result['processed'])
            if os.path.exists(zip_path):
                os.remove(zip_path)
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
