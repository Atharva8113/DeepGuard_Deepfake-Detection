import torch
from xception import FFPPXception
import os

def fix_state_dict_keys(state_dict):
    """
    Fix key mismatches between pretrained weights and model definition.
    Removes unwanted prefixes and renames final linear layer keys.
    """
    fixed_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("model.", "")
        # The final layer is stored as model.last_linear.1.{weight,bias}
        fixed_state_dict[k] = v
    return fixed_state_dict

def convert_pytorch_to_onnx(weight_path, onnx_output_path):
    print(f"📦 Loading model from: {weight_path}")
    model = FFPPXception()
    state_dict = torch.load(weight_path, map_location="cpu")

    # Fix key mismatches
    fixed_state_dict = fix_state_dict_keys(state_dict)

    # Load weights (handling Sequential block if needed)
    model_keys = model.state_dict().keys()
    if 'model.last_linear.1.weight' in state_dict:
        print("🔧 Detected Sequential-style last_linear layer")
        if isinstance(model.model.last_linear, torch.nn.Sequential):
            model.model.last_linear[0].weight.data = fixed_state_dict["last_linear.1.weight"]
            model.model.last_linear[0].bias.data = fixed_state_dict["last_linear.1.bias"]
        else:
            model.model.last_linear.weight.data = fixed_state_dict["last_linear.1.weight"]
            model.model.last_linear.bias.data = fixed_state_dict["last_linear.1.bias"]
    else:
        model.load_state_dict(fixed_state_dict)

    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, 3, 299, 299)

    # Export ONNX
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
    # Replace these paths as needed
    convert_pytorch_to_onnx("models/resnet50_best.pth", "models/resnet50_best.onnx")
    
