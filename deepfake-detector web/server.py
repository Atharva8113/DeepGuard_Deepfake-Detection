from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load ONNX models
MODEL_PATHS = {
    "vit_deepfake": "models/vit_deepfake.onnx",
    "xception_c23": "models/ffpp_c23.onnx",
    "xception_c40": "models/ffpp_c40.onnx",
    "resnet": "models/resnet50.onnx",  # your trained ResNet50
}

SESSIONS = {}
for name, path in MODEL_PATHS.items():
    print(f"🔍 Looking for model at: {path}")
    if os.path.exists(path):
        try:
            SESSIONS[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            print(f"✅ Loaded model: {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    else:
        print(f"❌ Model file not found: {path}")

# Preprocessing function
def preprocess_image(image_bytes, model_name):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    #  Different models expect different input sizes
    if model_name == "vit_deepfake":
        image = image.resize((224, 224))
    elif model_name == "resnet":
        image = image.resize((224, 224))  # ✅ ResNet50 expects 224x224
    else:
        image = image.resize((299, 299))  # Xception / EfficientNet defaults

    # Convert to numpy & normalize
    image_np = np.array(image).astype(np.float32) / 255.0

    # Standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std

    # Channel-first format [C, H, W]
    image_np = image_np.transpose(2, 0, 1)
    image_np = image_np[np.newaxis, :]  # add batch dimension

    return image_np.astype(np.float32)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes = request.files["image"].read()
    model_name = request.form.get("model", "resnet")  # default to resnet if not specified

    if model_name not in SESSIONS:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    try:
        input_tensor = preprocess_image(image_bytes, model_name)
        ort_session = SESSIONS[model_name]
        input_name = ort_session.get_inputs()[0].name

        # Run inference
        outputs = ort_session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_idx = int(np.argmax(probs))

        label = "real" if pred_idx == 1 else "fake"
        confidence = float(probs[pred_idx]) * 100

        print(f"🧠 Prediction: {label.upper()} ({confidence:.2f}%)")

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "probs": [round(float(p), 4) for p in probs]
        })

    except Exception as e:
        print("❌ Inference error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting DeepGuard Flask API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
