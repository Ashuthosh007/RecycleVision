
import argparse, os, json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from PIL import Image

def count_per_class(root):
    counts = {}
    for c in sorted(os.listdir(root)):
        cdir = os.path.join(root, c)
        if os.path.isdir(cdir):
            counts[c] = len([f for f in os.listdir(cdir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
    return counts

def show_examples(root, out_path="reports/sample_grid.png", per_class=3, size=(224,224)):
    # Make a simple grid image preview
    classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root,c))]
    images = []
    for c in classes:
        cdir = os.path.join(root, c)
        files = [f for f in os.listdir(cdir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        for f in files[:per_class]:
            images.append((c, os.path.join(cdir, f)))
    if not images:
        print("[eda] No images found to display.")
        return
    rows = len(classes)
    cols = per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if rows == 1: axes = [axes]
    for i, c in enumerate(classes):
        cimgs = [p for (cls, p) in images if cls==c]
        for j in range(cols):
            ax = axes[i][j] if rows>1 else axes[j]
            ax.axis("off")
            if j < len(cimgs):
                im = Image.open(cimgs[j]).convert("RGB").resize(size)
                ax.imshow(im)
                ax.set_title(c, fontsize=8)
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eda] Saved sample grid -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/train")
    ap.add_argument("--save_counts", type=str, default="reports/class_counts.json")
    args = ap.parse_args()

    os.makedirs("reports", exist_ok=True)
    counts = count_per_class(args.data_dir)
    print("[eda] Class counts:", counts)

    with open(args.save_counts, "w") as f:
        json.dump(counts, f, indent=2)

    show_examples(args.data_dir, out_path="reports/sample_grid.png", per_class=3)

if __name__ == "__main__":
    main()
