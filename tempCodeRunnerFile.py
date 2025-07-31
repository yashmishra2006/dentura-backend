from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import torch

# Init Flask app
app = Flask(__name__)
CORS(app)

# Load models
MODEL_PATHS = {
    "xray": "models/xray-yolo.pt",
    "gum": "models/gum-yolo.pt"
}

yolo_models = {
    model_type: YOLO(path)
    for model_type, path in MODEL_PATHS.items()
}

@app.route("/api/analyze/<model_type>", methods=["POST"])
def analyze(model_type):
    if model_type not in yolo_models:
        return jsonify({"error": f"Unsupported model type '{model_type}'"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = f"/tmp/{uuid.uuid4().hex}.jpg"
    image_file.save(image_path)

    try:
        # Run YOLO inference
        results = yolo_models[model_type](image_path)
        result = results[0]

        response = {
            "predictions": [],
            "processing_time": result.speed["inference"]
        }

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[cls_id]
            bbox = [float(x) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

            response["predictions"].append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(image_path)

