
import argparse, os, zipfile, random, shutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

random.seed(42)

def unzip_if_needed(zip_path: str, to_dir: str):
    os.makedirs(to_dir, exist_ok=True)
    # If to_dir already has folders, skip unzip
    if any(Path(to_dir).iterdir()):
        print(f"[data_prep] Skipping unzip: '{to_dir}' is not empty.")
        return
    print(f"[data_prep] Unzipping {zip_path} -> {to_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(to_dir)

def list_class_dirs(root: str) -> List[str]:
    return sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root,p))])

def gather_images(root: str) -> Dict[str, List[str]]:
    classes = list_class_dirs(root)
    images = defaultdict(list)
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in os.listdir(cdir):
            fpath = os.path.join(cdir, fn)
            if os.path.isfile(fpath) and fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                images[c].append(fpath)
    return images

def stratified_split(images_by_class: Dict[str, List[str]], val_split: float, test_split: float):
    train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)
    for c, paths in images_by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test[c] = paths[:n_test]
        val[c] = paths[n_test:n_test+n_val]
        train[c] = paths[n_test+n_val:]
    return train, val, test

def copy_split(split_dict, out_dir):
    for split_name, per_class in split_dict.items():
        for c, paths in tqdm(per_class.items(), desc=f"Copying {split_name}"):
            dst_dir = os.path.join(out_dir, split_name, c)
            os.makedirs(dst_dir, exist_ok=True)
            for src in paths:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=str, required=False, default=None, help="Path to dataset zip")
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Where to place unzipped dataset")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output split dir")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    if args.zip_path:
        unzip_if_needed(args.zip_path, args.raw_dir)

    # Assume after unzip, the actual dataset folder is the first with class subfolders; try to find it
    # If raw_dir has exactly one subfolder, use it; else, use raw_dir itself.
    candidates = [os.path.join(args.raw_dir, d) for d in os.listdir(args.raw_dir)]
    candidates = [d for d in candidates if os.path.isdir(d)]
    dataset_root = args.raw_dir
    if len(candidates) == 1:
        dataset_root = candidates[0]

    # If there is another nested single folder, descend once more
    subcands = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root,d))]
    if len(subcands) == 1 and len([p for p in os.listdir(subcands[0]) if os.path.isdir(os.path.join(subcands[0], p))])>0:
        dataset_root = subcands[0]

    classes = [p for p in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root,p))]
    if not classes:
        raise RuntimeError(f"No class folders found under {dataset_root}. Ensure the zip contains class-subfolder structure.")

    print(f"[data_prep] Detected classes: {sorted(classes)}")
    images_by_class = gather_images(dataset_root)
    train, val, test = stratified_split(images_by_class, args.val_split, args.test_split)

    # Clean out_dir
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    split_map = {
        "train": train,
        "val": val,
        "test": test
    }
    copy_split(split_map, args.out_dir)
    print("[data_prep] Done. Splits saved to:", args.out_dir)

if __name__ == "__main__":
    main()
