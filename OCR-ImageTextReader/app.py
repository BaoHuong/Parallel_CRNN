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

UPLOAD_FOLDER = os.path.basename('.')
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
        file = request.files['image']

        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)


        image = cv2.imread(f)
        os.remove(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)

        text,processing_time = sequent(filename)
        os.remove(filename)

    

        print(f"Processing time: {processing_time}s",flush=True)  # Debug print

        return jsonify({"text": text, "processing_time": processing_time})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run("0.0.0.0", port, threaded=True, debug=True)
