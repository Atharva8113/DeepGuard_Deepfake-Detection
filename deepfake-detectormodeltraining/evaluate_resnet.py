import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import ImageFolderDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_resnet50(checkpoint_path):
    """
    Load a trained ResNet50 checkpoint (handles 'model_state_dict' format).
    """
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Detect correct key automatically
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on the test dataset and calculate metrics.
    """
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of "real"
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, prec, rec, f1, auc


if __name__ == "__main__":
    # Paths
    dataset_path = "dataset"  # adjust if your folder name is different
    checkpoint_path = "models/resnet50_best.pth"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # Load test dataset
    from dataset import get_transforms

    _, val_tf = get_transforms(img_size=224)
    test_dataset = ImageFolderDataset(dataset_path, split="test", img_size=224, transforms=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load trained ResNet50 model
    print("📦 Loading trained ResNet50 model...")
    model = load_resnet50(checkpoint_path)

    # Evaluate
    print("⚙️ Evaluating on test set...")
    acc, prec, rec, f1, auc = evaluate_model(model, test_loader, device)

    print("\n✅ Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
