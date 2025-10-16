import tensorflow as tf
import tf2onnx
import os

def convert_h5_to_onnx(h5_path, onnx_output_path, input_shape=(224, 224, 3)):
    """
    Convert a Keras .h5 model into ONNX format.
    
    Args:
        h5_path (str): Path to the .h5 file
        onnx_output_path (str): Path to save the exported .onnx file
        input_shape (tuple): Expected input shape (H, W, C)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"❌ Model file not found: {h5_path}")

    print(f"📦 Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path)

    # Create a dummy input signature
    spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)

    # Convert using tf2onnx
    print("🔄 Converting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save the model
    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ ONNX model exported to: {onnx_output_path}")


if __name__ == "__main__":
    # Example usage:
    # For ResNet50 (224x224)
    convert_h5_to_onnx("models/tf_model.h5", "models/resnet50_deepfake.onnx", input_shape=(224, 224, 3))

    # For EfficientNet-B4 (380x380)
    #convert_h5_to_onnx("models/efficientnet_b4_deepfake.h5", "models/efficientnet_b4_deepfake.onnx", input_shape=(380, 380, 3))
