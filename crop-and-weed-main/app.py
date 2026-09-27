# app.py
import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2

# NEW: YOLO import
from ultralytics import YOLO

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/best.pt'   #  your trained model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# --- LOAD YOLO MODEL ---
try:
    model = YOLO(MODEL_PATH)
    print(" YOLO model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 🔥 UPDATED FUNCTION (YOLO instead of dummy mask)
def run_segmentation_model(image_path, model_name, sensitivity):
    try:
        # read image
        img = cv2.imread(image_path)

        # safety check
        if img is None:
            return None

        # confidence threshold (improved stability)
        threshold = max(0.3, int(sensitivity) / 100)

        # YOLO prediction (better accuracy with fixed size)
        results = model.predict(image_path, imgsz=340, verbose=False)

        names = model.names

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):

                if conf < threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                label_name = names[int(cls)]
                label = f"{label_name}: {conf:.2f}"

                # colors
                color = (0, 255, 0) if label_name == "crop" else (0, 0, 255)

                # skip very small detections (noise filter)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                # draw rectangle (thicker for visibility)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # text background box (fix visibility issue)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(img,
                              (x1, y1 - th - 6),
                              (x1 + tw, y1),
                              color,
                              -1)

                cv2.putText(img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1)

        # convert to RGB for web display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # encode base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return encoded_image

    except Exception as e:
        print(f"Error: {e}")
        return None
    
# --- ROUTES (UNCHANGED) ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment_image():
    file = request.files.get('file')
    model_name = request.form.get('model', 'adaptive_unet')
    sensitivity = request.form.get('sensitivity', 70)

    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = run_segmentation_model(filepath, model_name, sensitivity)

    os.remove(filepath)

    if result:
        return jsonify({'success': True, 'segmented_image': result})
    else:
        return jsonify({'error': 'Processing failed'}), 500


# --- RUN APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)