#!/usr/bin/env python3
"""
Test script for AI Background Remover API
Demonstrates how to use the API endpoints
"""

import requests
import base64
import json
import os
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:5000"  # Change to your Railway URL when deployed
# API_BASE_URL = "https://your-app.railway.app"  # Uncomment and update for production

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/models")
        print(f"Status: {response.status_code}")
        models = response.json()['models']
        print(f"Available models: {len(models)}")
        for model in models:
            print(f"  - {model['id']}: {model['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_upscale_methods():
    """Test upscale methods endpoint"""
    print("\n🔍 Testing upscale methods endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/upscale-methods")
        print(f"Status: {response.status_code}")
        methods = response.json()['methods']
        print(f"Available methods: {len(methods)}")
        for method in methods:
            print(f"  - {method['id']}: {method['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Upscale methods endpoint failed: {e}")
        return False

def test_file_upload(image_path):
    """Test file upload endpoint"""
    print(f"\n🔍 Testing file upload with {image_path}...")
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'isnet-general-use',
                'upscale': 'true',
                'scale_factor': '2.0',
                'upscale_method': 'lanczos',
                'padding': '10',
                'white_bg': 'false'
            }
            
            response = requests.post(f"{API_BASE_URL}/api/remove-background", 
                                   files=files, data=data)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            # Save result
            output_path = f"test_result_file.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Result saved to: {output_path}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        return False

def test_base64_api(image_path):
    """Test base64 API endpoint"""
    print(f"\n🔍 Testing base64 API with {image_path}...")
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare request
        payload = {
            'image': f'data:image/jpeg;base64,{image_data}',
            'model': 'isnet-general-use',
            'upscale': True,
            'scale_factor': 2.0,
            'upscale_method': 'lanczos',
            'padding': 10,
            'white_bg': False
        }
        
        response = requests.post(f"{API_BASE_URL}/api/remove-background-base64", 
                               json=payload)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                # Save result
                result_data = result['image'].split(',')[1]
                with open('test_result_base64.png', 'wb') as f:
                    f.write(base64.b64decode(result_data))
                print(f"✅ Result saved to: test_result_base64.png")
                print(f"Message: {result['message']}")
                return True
            else:
                print(f"❌ API returned success=false")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Base64 API failed: {e}")
        return False

def test_batch_processing(image_paths):
    """Test batch processing endpoint"""
    print(f"\n🔍 Testing batch processing with {len(image_paths)} images...")
    try:
        files = []
        for path in image_paths:
            if os.path.exists(path):
                files.append(('files', open(path, 'rb')))
        
        if not files:
            print("❌ No valid image files found")
            return False
        
        data = {
            'model': 'isnet-general-use',
            'upscale': 'true',
            'scale_factor': '2.0',
            'upscale_method': 'lanczos',
            'padding': '10',
            'white_bg': 'false'
        }
        
        response = requests.post(f"{API_BASE_URL}/api/batch-remove", 
                               files=files, data=data)
        
        # Close files
        for _, file in files:
            file.close()
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            # Save result
            with open('test_batch_results.zip', 'wb') as f:
                f.write(response.content)
            print(f"✅ Batch results saved to: test_batch_results.zip")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Background Remover API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health()
    models_ok = test_models()
    methods_ok = test_upscale_methods()
    
    if not all([health_ok, models_ok, methods_ok]):
        print("\n❌ Basic endpoints failed. Make sure the server is running.")
        return
    
    # Test with actual image processing
    test_image = "5-Amino-1mg-Lyophilized-Synthetic-Research-Peptide-2.webp"
    
    if os.path.exists(test_image):
        print(f"\n📸 Testing with image: {test_image}")
        
        # Test file upload
        file_ok = test_file_upload(test_image)
        
        # Test base64 API
        base64_ok = test_base64_api(test_image)
        
        # Test batch processing
        batch_ok = test_batch_processing([test_image])
        
        print("\n📊 Test Results Summary:")
        print(f"  Health Check: {'✅' if health_ok else '❌'}")
        print(f"  Models API: {'✅' if models_ok else '❌'}")
        print(f"  Methods API: {'✅' if methods_ok else '❌'}")
        print(f"  File Upload: {'✅' if file_ok else '❌'}")
        print(f"  Base64 API: {'✅' if base64_ok else '❌'}")
        print(f"  Batch Processing: {'✅' if batch_ok else '❌'}")
        
        if all([health_ok, models_ok, methods_ok, file_ok, base64_ok, batch_ok]):
            print("\n🎉 All tests passed! API is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above.")
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("   Please ensure you have an image file to test with.")

if __name__ == "__main__":
    main()
