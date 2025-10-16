import torch
import torch.nn as nn
from torchvision import models

def load_resnet50(checkpoint_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

if __name__ == "__main__":
    checkpoint_path = "models/resnet50_best.pth"
    onnx_output_path = "models/resnet50.onnx"

    model = load_resnet50(checkpoint_path)

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"✅ ResNet50 model exported to: {onnx_output_path}")
