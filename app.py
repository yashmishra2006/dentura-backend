import os
import uuid
import requests
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import traceback
import gdown

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Ensure you've set your API key in your environment variables.

# Define the endpoint for Gemini NLP (or any API you're using)
GEMINI_API_URL = "https://gemini.googleapis.com/v1/detectIntent"

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# === Load YOLOv8 X-ray Model ===
YOLO_MODELS = {}
YOLO_MODEL_PATH = "yolov8.pt"
RESNET_MODEL_PATH = "resnet_gums.pth"
# Initialize YOLO model with error handling
try:
    if not os.path.exists(YOLO_MODEL_PATH):
        print("⬇️ Downloading YOLO model...")
        gdown.download(id="1_AaiymL13F-fxaElD_tQRiImfJ4iLGz_", output=YOLO_MODEL_PATH, quiet=False)
        YOLO_MODELS["xray"] = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO X-ray model loaded successfully.")
    else:
        print("❌ YOLO model file not found. Please ensure 'models/xray-yolo.pt' exists.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")

# === Load ResNet Gums Classifier ===
NUM_CLASSES = 6
CLASSES = ["Calculus", "Caries", "Gingivitis", "Mouth Ulcer", "Tooth Discoloration", "Hypodontia"]

# Load the ResNet model
resnet_model = None
try:
    if not os.path.exists("resnet_gums.pth"):
        print("⬇️ Downloading model...")
        gdown.download(id="1XemIV3DEG7ZFvlH-u-a56fZ3h2a_Xrsa", output=RESNET_MODEL_PATH, quiet=False)
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
        resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location="cpu"))
        resnet_model.eval()
        print("✅ ResNet gums model loaded successfully.")
    else:
        print("❌ ResNet model file not found. Please ensure 'models/resnet_gums.pth' exists.")
except Exception as e:
    print(f"❌ Failed to load ResNet model: {e}")

# Define severity mapping for YOLO model classes
severity_mapping_yolo = {
    "Caries": "moderate",
    "Crown": "moderate",
    "Filling": "moderate",
    "Implant": "high",
    "Malaligned": "low",
    "Mandibular Canal": "high",
    "Missing teeth": "high",
    "Periapical lesion": "high",
    "Retained root": "high",
    "Root Canal Treatment": "moderate",
    "Root Piece": "high",
    "Impacted tooth": "high",
    "Maxillary sinus": "moderate",
    "Bone Loss": "high",
    "Fracture teeth": "high",
    "Permanent Teeth": "moderate",
    "Supra Eruption": "moderate",
    "TAD": "low",
    "Abutment": "moderate",
    "Attrition": "moderate",
    "Bone defect": "high",
    "Gingival former": "low",
    "Metal band": "low",
    "Orthodontic brackets": "low",
    "Permanent retainer": "low",
    "Post-core": "moderate",
    "Plating": "moderate",
    "Wire": "low",
    "Cyst": "high",
    "Root resorption": "high",
    "Primary teeth": "low"
}

# Define severity mapping for ResNet model classes
severity_mapping_resnet = {
    "Calculus": "low",
    "Caries": "moderate",
    "Gingivitis": "moderate",
    "Mouth Ulcer": "low",
    "Tooth Discoloration": "low",
    "Hypodontia": "high"
}

# === Image Preprocessor for ResNet ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Helper function to save image temporarily
def save_temp_image(file_storage):
    temp_dir = './temp_images'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(temp_dir, filename)
    
    try:
        file_storage.save(path)
        return path
    except Exception as e:
        print(f"Error saving temporary image: {e}")
        raise

# Helper function to convert image to base64 string
def image_to_base64(img: Image.Image) -> str:
    try:
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

# === Color mapping for different dental conditions ===
COLOR_MAPPING = {
    "Caries": "#FF0000",  # Red
    "Crown": "#00FF00",  # Green
    "Filling": "#0000FF",  # Blue
    "Implant": "#FF00FF",  # Magenta
    "Malaligned": "#FFFF00",  # Yellow
    "Mandibular Canal": "#00FFFF",  # Cyan
    "Missing teeth": "#FFA500",  # Orange
    "Periapical lesion": "#800080",  # Purple
    "Retained root": "#FFC0CB",  # Pink
    "Root Canal Treatment": "#A52A2A",  # Brown
    "Root Piece": "#808080",  # Gray
    "Impacted tooth": "#FF69B4",  # Hot Pink
    "Maxillary sinus": "#32CD32",  # Lime Green
    "Bone Loss": "#DC143C",  # Crimson
    "Fracture teeth": "#B22222",  # Fire Brick
    "Permanent Teeth": "#228B22",  # Forest Green
    "Supra Eruption": "#FFD700",  # Gold
    "TAD": "#4169E1",  # Royal Blue
    "Abutment": "#8A2BE2",  # Blue Violet
    "Attrition": "#D2691E",  # Chocolate
    "Bone defect": "#FF4500",  # Orange Red
    "Gingival former": "#9ACD32",  # Yellow Green
    "Metal band": "#708090",  # Slate Gray
    "Orthodontic brackets": "#20B2AA",  # Light Sea Green
    "Permanent retainer": "#87CEEB",  # Sky Blue
    "Post-core": "#DDA0DD",  # Plum
    "Plating": "#98FB98",  # Pale Green
    "Wire": "#F0E68C",  # Khaki
    "Cyst": "#8B0000",  # Dark Red
    "Root resorption": "#483D8B",  # Dark Slate Blue
    "Primary teeth": "#00CED1"  # Dark Turquoise
}

# Helper function to draw clean color-coded boxes only
def draw_enhanced_annotations(image_path, predictions):
    try:
        original_image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = original_image.size
        
        # Resize image to make it bigger (2x size) for better visibility
        scale_factor = 2.0
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Resize the image
        annotated_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(annotated_image)
        
        # Load font for legend only
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            font_large = ImageFont.truetype("arial.ttf", 30)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)  # macOS
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 30)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)  # Linux
                    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
                except:
                    font = ImageFont.load_default()
                    font_large = ImageFont.load_default()

        # Sort predictions by area (smaller boxes first) so small boxes are visible
        sorted_predictions = sorted(predictions, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
        
        for i, prediction in enumerate(sorted_predictions):
            bbox = prediction["bbox"]
            class_name = prediction["class"]
            
            # Scale bbox coordinates to match resized image
            scaled_bbox = [
                bbox[0] * scale_factor,
                bbox[1] * scale_factor,
                bbox[2] * scale_factor,
                bbox[3] * scale_factor
            ]
            
            # Get color for this class, default to red if not found
            color = COLOR_MAPPING.get(class_name, "#FF0000")
            
            # Convert hex color to RGB tuple
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw only the bounding box with thick, colored lines
            line_width = 5  # Even thicker for better visibility
            draw.rectangle([scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]], 
                         outline=color_rgb, width=line_width)

        # Add comprehensive legend at the bottom of the image
        legend_height = 250  # Increased for more detailed legend
        img_width, img_height = annotated_image.size
        
        # Create a new image with space for legend
        legend_image = Image.new('RGB', (img_width, img_height + legend_height), 'white')
        legend_image.paste(annotated_image, (0, 0))
        
        # Draw legend
        legend_draw = ImageDraw.Draw(legend_image)
        legend_y_start = img_height + 20
        
        # Get unique classes from predictions with their confidence scores
        class_info = {}
        for pred in predictions:
            class_name = pred["class"]
            confidence = pred["confidence"]
            if class_name not in class_info or confidence > class_info[class_name]["confidence"]:
                class_info[class_name] = {
                    "confidence": confidence,
                    "count": sum(1 for p in predictions if p["class"] == class_name)
                }
        
        # Sort by class name alphabetically
        sorted_classes = sorted(class_info.keys())
        
        # Draw legend title
        title = f"Detected Conditions ({len(sorted_classes)} unique types, {len(predictions)} total detections):"
        legend_draw.text((20, legend_y_start), title, fill="black", font=font_large)
        
        # Draw legend items in columns with detailed info
        cols = 3  # Fewer columns for more detailed info
        col_width = img_width // cols
        
        for i, class_name in enumerate(sorted_classes):
            col = i % cols
            row = i // cols
            
            x = 20 + col * col_width
            y = legend_y_start + 50 + row * 35  # More vertical spacing
            
            color = COLOR_MAPPING.get(class_name, "#FF0000")
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw larger color box with border
            box_size = 25
            legend_draw.rectangle([x, y, x + box_size, y + box_size], 
                                fill=color_rgb, outline="black", width=2)
            
            # Draw detailed class information
            info = class_info[class_name]
            detail_text = f"{class_name} (×{info['count']}) - {info['confidence']:.2f}"
            legend_draw.text((x + box_size + 10, y + 3), detail_text, fill="black", font=font)

        return legend_image
        
    except Exception as e:
        print(f"Error drawing enhanced annotations: {e}")
        # Return original image if annotation fails
        return Image.open(image_path).convert("RGB")

# === Enhanced YOLO Analysis Route (X-ray) ===
@app.route("/api/analyze/xray", methods=["POST"])
def analyze_xray():
    if "xray" not in YOLO_MODELS:
        return jsonify({"error": "YOLO X-ray model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    image_path = None
    try:
        image_path = save_temp_image(image_file)
        
        # Run YOLO inference
        results = YOLO_MODELS["xray"](image_path)
        result = results[0]

        all_predictions = []  # Keep all predictions for annotations
        unique_classes = {}   # Keep track of unique classes for response
        
        # Process predictions if any boxes are detected
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[cls_id]
                bbox = [float(x) for x in box.xyxy[0].tolist()]

                prediction = {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "color": COLOR_MAPPING.get(class_name, "#FF0000")
                }

                all_predictions.append(prediction)
                
                # Keep only the highest confidence prediction for each class for the response
                if class_name not in unique_classes or confidence > unique_classes[class_name]["confidence"]:
                    unique_classes[class_name] = prediction

        # Create annotated image with ALL predictions (not just unique ones)
        if all_predictions:
            annotated_image = draw_enhanced_annotations(image_path, all_predictions)
            # Get the prediction with highest confidence for severity mapping
            primary_prediction = max(unique_classes.values(), key=lambda x: x['confidence'])
            detected_disease = primary_prediction["class"]
            severity = severity_mapping_yolo.get(detected_disease, "low")
        else:
            # No detections found
            annotated_image = Image.open(image_path).convert("RGB")
            detected_disease = "No significant findings"
            severity = "low"

        # Convert annotated image to base64
        annotated_image_str = image_to_base64(annotated_image)

        # Prepare response - send unique classes but annotate everything
        unique_predictions = list(unique_classes.values())
        
        response_data = {
            "predictions": unique_predictions,  # Unique classes for frontend processing
            "all_detections": all_predictions,  # All detections for reference
            "unique_classes_count": len(unique_predictions),
            "total_detections": len(all_predictions),
            "severity": severity,
            "color_mapping": COLOR_MAPPING,  # Send color mapping to frontend
            "processing_time": result.speed.get("inference", 0.0) if hasattr(result, 'speed') else 0.0,
            "annotated_image": annotated_image_str
        }

        print(f"X-ray analysis completed: {len(unique_predictions)} unique classes, {len(all_predictions)} total detections, severity: {severity}")
        return jsonify(response_data)

    except Exception as e:
        print(f"X-ray analysis error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error removing temporary file: {e}")

# === Gums Analysis Route (ResNet) ===
@app.route("/api/analyze/gum", methods=["POST"])
def analyze_gum():
    if resnet_model is None:
        return jsonify({"error": "ResNet gums model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    image_path = None
    try:
        image_path = save_temp_image(image_file)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = resnet_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_index].item()

            # Map the predicted class index to the class name
            predicted_class_name = CLASSES[predicted_class_index]

        # Set severity based on the detected disease
        severity = severity_mapping_resnet.get(predicted_class_name, "low")
        
        response_data = {
            "prediction": predicted_class_name,
            "confidence": round(confidence, 4),
            "severity": severity
        }

        print(f"Gum analysis completed - Class: {predicted_class_name}, Confidence: {confidence:.4f}, Severity: {severity}")
        return jsonify(response_data)

    except Exception as e:
        print(f"Gum analysis error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error removing temporary file: {e}")

# === Treatment Recommendations Generator ===
def generate_treatment_recommendations(disease, severity):
    """Generate treatment recommendations based on disease and severity"""
    
    base_recommendations = {
        "Caries": {
            "low": "1. Schedule routine dental cleaning\n2. Use fluoride toothpaste daily\n3. Reduce sugar intake\n4. Follow up in 6 months",
            "moderate": "1. Schedule immediate dental examination\n2. Professional fluoride treatment\n3. Consider dental fillings\n4. Improve oral hygiene routine\n5. Follow up in 3 months",
            "high": "1. Urgent dental consultation required\n2. Immediate restorative treatment\n3. Possible root canal therapy\n4. Pain management if needed\n5. Comprehensive oral health assessment"
        },
        "Gingivitis": {
            "low": "1. Improve daily brushing technique\n2. Use antimicrobial mouthwash\n3. Regular flossing\n4. Professional cleaning every 6 months",
            "moderate": "1. Professional deep cleaning (scaling)\n2. Antibiotic therapy if recommended\n3. Improved oral hygiene routine\n4. Follow up in 2-3 months",
            "high": "1. Immediate periodontal treatment\n2. Deep scaling and root planing\n3. Possible antibiotic treatment\n4. Strict oral hygiene protocol\n5. Monthly follow-ups initially"
        },
        "Calculus": {
            "low": "1. Professional dental cleaning\n2. Regular brushing and flossing\n3. Use tartar control toothpaste\n4. 6-month dental checkups",
            "moderate": "1. Professional scaling and polishing\n2. Improved oral hygiene education\n3. More frequent cleanings (every 4 months)\n4. Consider electric toothbrush",
            "high": "1. Comprehensive periodontal therapy\n2. Multiple deep cleaning sessions\n3. Specialized periodontal instruments\n4. Monthly maintenance initially"
        },
        "Mouth Ulcer": {
            "low": "1. Use oral pain relief gel\n2. Avoid spicy and acidic foods\n3. Maintain oral hygiene\n4. Monitor for healing (7-10 days)",
            "moderate": "1. Prescription antimicrobial mouthwash\n2. Topical corticosteroids if recommended\n3. Avoid irritating foods\n4. Dental consultation if persists",
            "high": "1. Immediate dental/medical evaluation\n2. Possible biopsy if indicated\n3. Prescription medications\n4. Rule out systemic conditions"
        },
        "Tooth Discoloration": {
            "low": "1. Professional dental cleaning\n2. Whitening toothpaste\n3. Avoid staining substances\n4. Regular dental checkups",
            "moderate": "1. Professional teeth whitening\n2. Dental examination for underlying causes\n3. Improved oral hygiene\n4. Consider cosmetic options",
            "high": "1. Comprehensive dental evaluation\n2. Professional whitening treatment\n3. Possible veneers or crowns\n4. Address underlying dental issues"
        },
        "Hypodontia": {
            "low": "1. Regular monitoring\n2. Maintain existing teeth health\n3. Orthodontic consultation\n4. Consider space maintenance",
            "moderate": "1. Orthodontic evaluation\n2. Space management planning\n3. Consider partial dentures\n4. Regular dental monitoring",
            "high": "1. Comprehensive treatment planning\n2. Multidisciplinary approach\n3. Implant consultation\n4. Prosthetic rehabilitation options"
        },
        "Missing teeth": {
            "low": "1. Maintain adjacent teeth health\n2. Consider dental implant\n3. Regular dental checkups\n4. Prevent adjacent teeth shifting",
            "moderate": "1. Dental implant consultation\n2. Bridge or partial denture options\n3. Bone density evaluation\n4. Orthodontic assessment",
            "high": "1. Immediate prosthetic evaluation\n2. Full mouth rehabilitation planning\n3. Implant or denture options\n4. Nutritional counseling"
        },
        "Bone Loss": {
            "low": "1. Improve oral hygiene\n2. Regular periodontal maintenance\n3. Calcium and vitamin D supplements\n4. Professional cleanings every 3-4 months",
            "moderate": "1. Periodontal therapy\n2. Bone grafting evaluation\n3. Antibiotic therapy if needed\n4. Close monitoring every 3 months",
            "high": "1. Immediate periodontal specialist referral\n2. Advanced periodontal therapy\n3. Possible bone regeneration procedures\n4. Frequent maintenance therapy"
        }
    }
    
    # Default recommendations if disease not found
    default_recommendations = {
        "low": f"1. Monitor {disease} condition\n2. Maintain good oral hygiene\n3. Regular dental checkups\n4. Follow preventive care guidelines",
        "moderate": f"1. Schedule dental consultation for {disease}\n2. Follow professional treatment plan\n3. Improve oral care routine\n4. Regular monitoring",
        "high": f"1. Seek immediate dental attention for {disease}\n2. Follow urgent treatment protocol\n3. Pain management if needed\n4. Comprehensive follow-up care"
    }
    
    disease_recs = base_recommendations.get(disease, default_recommendations)
    return disease_recs.get(severity, disease_recs.get("moderate", f"Consult dental professional for {disease} treatment"))

# === ChatGPT/AI Recommendations Route ===
@app.route("/api/recommendations/chatgpt", methods=["POST"])
def chatgpt_recommendations():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        disease = data.get("disease", "").strip()
        severity = data.get("severity", "").strip()
        context = data.get("context", "").strip()

        if not disease or not severity:
            return jsonify({"error": "Missing required fields: disease and severity"}), 400

        # For now, use the built-in recommendation generator
        # You can replace this with actual AI API calls later
        recommendations = generate_treatment_recommendations(disease, severity)

        # If you want to use an actual AI API, uncomment and modify this section:
        """
        if GOOGLE_API_KEY:
            try:
                prompt_message = f"As a dental professional, provide treatment recommendations for a patient diagnosed with {disease} with {severity} severity. Context: {context}. Please provide specific, actionable treatment advice."
                
                # Example API call structure (modify based on your AI service)
                payload = {
                    "key": GOOGLE_API_KEY,
                    "input_text": prompt_message
                }
                
                response = requests.post(GEMINI_API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    ai_recommendations = result.get("response", recommendations)
                    recommendations = ai_recommendations
                else:
                    print(f"AI API failed with status {response.status_code}, using fallback")
            except Exception as ai_error:
                print(f"AI API error: {ai_error}, using fallback recommendations")
        """

        response_data = {
            "recommendations": recommendations
        }

        print(f"Generated recommendations for {disease} ({severity})")
        return jsonify(response_data)

    except Exception as e:
        print(f"Recommendation generation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500

# === Health Check Route ===
@app.route("/api/health", methods=["GET"])
def health_check():
    models_status = {
        "yolo_xray": "xray" in YOLO_MODELS,
        "resnet_gums": resnet_model is not None
    }
    
    return jsonify({
        "status": "healthy",
        "models": models_status,
        "timestamp": str(uuid.uuid4())
    })

# === Model Info Route ===
@app.route("/api/models/info", methods=["GET"])
def model_info():
    info = {
        "yolo_xray": {
            "loaded": "xray" in YOLO_MODELS,
            "classes": list(severity_mapping_yolo.keys()) if "xray" in YOLO_MODELS else []
        },
        "resnet_gums": {
            "loaded": resnet_model is not None,
            "classes": CLASSES if resnet_model is not None else []
        }
    }
    return jsonify(info)

# === Error Handlers ===
@app.errorhandler(500)
def handle_500(e):
    print(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error occurred"}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def handle_413(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(400)
def handle_400(e):
    return jsonify({"error": "Bad request"}), 400

# === Configuration ===
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

if __name__ == "__main__":
    print("🚀 Starting Dentura Backend Server...")
    print(f"✅ YOLO Model Status: {'Loaded' if 'xray' in YOLO_MODELS else 'Not Loaded'}")
    print(f"✅ ResNet Model Status: {'Loaded' if resnet_model is not None else 'Not Loaded'}")
    print("📡 Server starting on http://127.0.0.1:5000")
    
    # Create temp directory if it doesn't exist
    if not os.path.exists('./temp_images'):
        os.makedirs('./temp_images')
        print("📁 Created temp_images directory")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)