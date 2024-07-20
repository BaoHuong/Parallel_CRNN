import sys
sys.path.append(r'C:\Users\HP PC\OneDrive\Desktop\Asset Game\ocr\Parallel_CRNN\OCR-ImageTextReader\backend\crnn-pytorch\src')
from predict import processing
import os
import cv2
import pytesseract
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from PIL import Image
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

app = Flask(__name__, static_url_path='', static_folder='frontend/build')
CORS(app, support_credentials=True)

UPLOAD_FOLDER = r'C:\Users\HP PC\OneDrive\Desktop\Asset Game\ocr\Parallel_CRNN\OCR-ImageTextReader\backend\crnn-pytorch\demo'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def sequent(fileName):
    start_time = time.time()  # Record start time
    text = pytesseract.image_to_string(Image.open(fileName))
    end_time = time.time()
    process = end_time - start_time
    return text, process
    

@app.route("/", defaults={'path': ''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/api/ocr', methods=['POST', 'GET'])
def upload_file():
    if request.method == "GET":
        return "Nothing to return"
    elif request.method == "POST":
        try:
            file = request.files['image']
            
            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            # Define the path for saving the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving file to: {file_path}", flush=True)
            file.save(file_path)
            print(f"File saved: {os.path.exists(file_path)}", flush=True)
            base_directory = r'C:\Users\HP PC\OneDrive\Desktop\Asset Game\ocr\Parallel_CRNN\OCR-ImageTextReader\backend\crnn-pytorch'
            relative_path = os.path.relpath(file_path, start=base_directory)
            windows_path = '.\\backend\\crnn-pytorch\\' + relative_path.replace('/', '\\')
            start_processing = time.time()
            path_list = processing([windows_path])
            end_processing = time.time()

            print(path_list)
            processing_time = end_processing - start_processing

            print(f"Processing time: {processing_time}s", flush=True)  # Debug print

            return jsonify({"text": path_list, "processing_time": processing_time})

        except Exception as e:
            print(f"An error occurred: {e}", flush=True)
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run("0.0.0.0", port, threaded=True, debug=True)