'''from flask import Flask, request, jsonify
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import io

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("models/ffpp_c23.onnx")
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(".", filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_tensor})

    score = float(result[0][0][0])
    prediction = "real" if score > 0.5 else "fake"

    return jsonify({
        "prediction": prediction,
        "confidence": score * 100
    })

if __name__ == "__main__":
    app.run(debug=True)

'''
import h5py

with h5py.File("models/tf_model.h5", "r") as f:
    print(f.keys())          # top-level keys
    if "model_config" in f:
        print("✅ Full model config exists")
    else:
        print("⚠️ Only weights saved")
