# dataset.py
import os, glob
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224, transforms=None):
        self.paths = []
        self.labels = []
        for label, cls in enumerate(['fake','real']):  # 0 fake, 1 real
            p = os.path.join(root_dir, split, cls)
            files = glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png"))
            self.paths += files
            self.labels += [label] * len(files)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}/{split}")
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, self.labels[idx]

def get_transforms(img_size=224):
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    return train_tf, val_tf
