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
from background_remover import BackgroundRemover

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize background remover
try:
    bg_remover = BackgroundRemover()
    logger.info("Background remover initialized successfully")
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
        model = request.form.get('model', 'isnet-general-use')
        upscale = request.form.get('upscale', 'false').lower() == 'true'
        scale_factor = float(request.form.get('scale_factor', '2.0'))
        upscale_method = request.form.get('upscale_method', 'lanczos')
        padding = int(request.form.get('padding', '10'))
        white_bg = request.form.get('white_bg', 'false').lower() == 'true'
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        input_path = f"temp_{filename}"
        file.save(input_path)
        
        try:
            # Process the image
            if white_bg:
                result_path = bg_remover.remove_background_with_white_bg(
                    input_path, 
                    model=model,
                    auto_crop=True,
                    padding=padding,
                    upscale=upscale,
                    scale_factor=scale_factor,
                    upscale_method=upscale_method
                )
            else:
                result_path = bg_remover.remove_background(
                    input_path,
                    model=model,
                    auto_crop=True,
                    padding=padding,
                    upscale=upscale,
                    scale_factor=scale_factor,
                    upscale_method=upscale_method
                )
            
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
        model = request.form.get('model', 'isnet-general-use')
        upscale = request.form.get('upscale', 'false').lower() == 'true'
        scale_factor = float(request.form.get('scale_factor', '2.0'))
        upscale_method = request.form.get('upscale_method', 'lanczos')
        padding = int(request.form.get('padding', '10'))
        white_bg = request.form.get('white_bg', 'false').lower() == 'true'
        
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
                            model=model,
                            auto_crop=True,
                            padding=padding,
                            upscale=upscale,
                            scale_factor=scale_factor,
                            upscale_method=upscale_method
                        )
                    else:
                        result_path = bg_remover.remove_background(
                            input_path,
                            model=model,
                            auto_crop=True,
                            padding=padding,
                            upscale=upscale,
                            scale_factor=scale_factor,
                            upscale_method=upscale_method
                        )
                    
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
    app.run(host='0.0.0.0', port=port, debug=False)
