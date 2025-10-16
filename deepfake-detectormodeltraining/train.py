# train.py
import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from dataset import ImageFolderDataset, get_transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses, preds, targets = [], [], []
    for images, labels in tqdm(loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        preds.extend(pred.tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())
    return np.mean(losses), accuracy_score(targets, preds)

def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets, probs_list = [], [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            prob = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
            pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            probs_list.extend(prob.tolist())
            preds.extend(pred.tolist())
            targets.extend(labels.detach().cpu().numpy().tolist())
    auc = roc_auc_score(targets, probs_list) if len(set(targets))>1 else 0.0
    return np.mean(losses), accuracy_score(targets, preds), auc

def load_state_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    try:
        model.load_state_dict(state)
    except Exception as e:
        model.load_state_dict(state, strict=False)
        print("Warning: loaded with strict=False", e)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    default_sizes = {"resnet50":224, "efficientnet_b4":380}
    if args.img_size is None:
        args.img_size = default_sizes.get(args.model_name, 224)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf, val_tf = get_transforms(img_size=args.img_size)
    train_ds = ImageFolderDataset(args.data, split='train', img_size=args.img_size, transforms=train_tf)
    val_ds = ImageFolderDataset(args.data, split='val', img_size=args.img_size, transforms=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device=="cuda" else None

    start_epoch = 1
    best_auc = 0.0
    if args.resume:
        model = load_state_into_model(model, args.resume)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        v_loss, v_acc, v_auc = validate(model, val_loader, criterion, device)
        print(f"Train loss {t_loss:.4f} acc {t_acc:.4f} | Val loss {v_loss:.4f} acc {v_acc:.4f} auc {v_auc:.4f}")
        scheduler.step()
        if v_auc > best_auc:
            best_auc = v_auc
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'auc': best_auc}
            torch.save(ckpt, os.path.join(args.out_dir, f"{args.model_name}_best.pth"))
            print("Saved best checkpoint")
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.out_dir, f"{args.model_name}_last.pth"))
    print("Training finished. Best AUC:", best_auc)

if __name__ == "__main__":
    main()
