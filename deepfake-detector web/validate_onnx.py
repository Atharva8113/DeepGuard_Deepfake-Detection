import onnx
import onnxruntime as ort
import numpy as np

def validate_onnx_model(onnx_path):
    # Step 1: Load and check ONNX format
    print("🔍 Checking ONNX model format...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model structure is valid.")

    # Step 2: Create ONNX Runtime session
    print("🧠 Initializing ONNX Runtime session...")
    ort_session = ort.InferenceSession(onnx_path)

    # Step 3: Prepare dummy input
    dummy_input = np.random.randn(1, 3, 299, 299).astype(np.float32)

    # Step 4: Run inference
    print("🚀 Running inference...")
    outputs = ort_session.run(None, {"input": dummy_input})

    print("✅ Inference successful!")
    print("Output:", outputs[0])
    print("Output shape:", outputs[0].shape)

if __name__ == "__main__":
    validate_onnx_model("models/ffpp_c23.onnx")


'''import torch
from xception import FFPPXception
import os

def fix_state_dict_keys(state_dict):
    """
    Fix key mismatches between pretrained weights and model definition.
    Removes unwanted prefixes and renames final linear layer keys.
    """
    fixed_state_dict = {}
    for k, v in state_dict.items():
        # Remove common prefixes
        k = k.replace("module.", "")
        k = k.replace("model.", "")

        # Fix final layer name mismatch
        k = k.replace("last_linear.1.", "last_linear.")

        fixed_state_dict[k] = v
    return fixed_state_dict

def convert_pytorch_to_onnx(weight_path, onnx_output_path):
    model = FFPPXception()  # Initialize your model
    state_dict = torch.load(weight_path, map_location="cpu")

    # Fix key mismatches
    fixed_state_dict = fix_state_dict_keys(state_dict)

    # Load weights into actual model
    model.model.load_state_dict(fixed_state_dict)

    # Set model to eval mode
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 299, 299)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"✅ ONNX model exported to: {onnx_output_path}")

if __name__ == "__main__":
    convert_pytorch_to_onnx("models/ffpp_c40.pth", "models/ffpp_c40.onnx")
'''