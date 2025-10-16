# prepare_data.py
import os, glob, random, shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

def extract_and_crop(video_path, out_dir, mtcnn, img_size=224, frames_per_video=50, min_prob=0.90):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release(); return 0
    step = max(1, total // frames_per_video)
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                best_i = int(probs.argmax()) if len(probs) > 1 else 0
                if probs[best_i] >= min_prob:
                    x1, y1, x2, y2 = [int(v) for v in boxes[best_i]]
                    x1, y1 = max(0,x1), max(0,y1)
                    x2, y2 = min(img.width,x2), min(img.height,y2)
                    if x2-x1>10 and y2-y1>10:
                        face = img.crop((x1,y1,x2,y2)).resize((img_size, img_size))
                        out_path = os.path.join(out_dir, f"{Path(video_path).stem}_{idx}.jpg")
                        face.save(out_path)
                        saved += 1
        idx += 1
    cap.release()
    return saved

def process_videos(videos_dir, out_dir, label, img_size=224, frames_per_video=50, sample=None):
    os.makedirs(out_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=True, device='cpu')
    videos = glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True) + glob.glob(os.path.join(videos_dir, "**", "*.mov"), recursive=True)
    if sample:
        videos = random.sample(videos, min(len(videos), sample))
    total_saved = 0
    for v in tqdm(videos, desc=f"Processing {label}"):
        saved = extract_and_crop(v, out_dir, mtcnn, img_size, frames_per_video)
        total_saved += saved
    return total_saved

def make_splits(real_videos, fake_videos, out_root="dataset", img_size=224, frames=50, train_ratio=0.8, val_ratio=0.1):
    tmp = os.path.join(out_root, "tmp_extract")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    print("Extracting real videos...")
    r_out = os.path.join(tmp, "real")
    process_videos(real_videos, r_out, "real", img_size, frames)
    print("Extracting fake videos...")
    f_out = os.path.join(tmp, "fake")
    process_videos(fake_videos, f_out, "fake", img_size, frames)

    for label in ["real","fake"]:
        imgs = sorted(glob.glob(os.path.join(tmp, label, "*.jpg")))
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = imgs[:n_train]
        val = imgs[n_train:n_train+n_val]
        test = imgs[n_train+n_val:]
        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest = os.path.join(out_root, split_name, label)
            os.makedirs(dest, exist_ok=True)
            for p in split_files:
                shutil.copy(p, dest)

    print("Done. Dataset at:", out_root)
    shutil.rmtree(tmp)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_videos", required=True)
    parser.add_argument("--fake_videos", required=True)
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    make_splits(args.real_videos, args.fake_videos, out_root=args.out, img_size=args.img_size, frames=args.frames)
