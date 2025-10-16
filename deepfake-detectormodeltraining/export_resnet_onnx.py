import torch
import torch.nn as nn
from torchvision import models

def load_resnet50(checkpoint_path):
    """
    Load the trained ResNet50 model checkpoint (handles model_state_dict formats)
    """
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Detect proper key for state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    checkpoint_path = "models/resnet50_best.pth"   # <-- your trained weights
    onnx_output_path = "models/resnet50.onnx"      # <-- output path

    print("📦 Loading model checkpoint...")
    model = load_resnet50(checkpoint_path)

    dummy_input = torch.randn(1, 3, 224, 224)  # 224x224 input image

    print("🔄 Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"✅ Export complete! Saved at: {onnx_output_path}")
