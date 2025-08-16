import argparse, os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from config import Config


def make_dataset(dir_path: str, img_size: int, batch_size: int, seed: int):
    ds = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        label_mode="categorical",
    )
    return ds


def plot_confusion_matrix(
    cm, classes, normalize=False, out_path="reports/confusion_matrix.png"
):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")
    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved confusion matrix -> {out_path}")


def load_model_auto(weights_path):
    """Auto-load model based on file extension or SavedModel format."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")

    # Append .keras if no extension given
    if not (
        weights_path.endswith(".keras") or weights_path.endswith(".h5")
    ) and os.path.isfile(weights_path + ".keras"):
        weights_path += ".keras"

    try:
        print(f"[evaluate] Trying to load model from {weights_path} ...")
        model = tf.keras.models.load_model(weights_path)
        print("[evaluate] Model loaded successfully with tf.keras.models.load_model")
        return model
    except Exception as e:
        print(f"[evaluate] Standard load failed: {e}")

    # Try TFSMLayer for SavedModel format
    if os.path.isdir(weights_path):
        try:
            print(
                f"[evaluate] Attempting to load as TensorFlow SavedModel from {weights_path} ..."
            )
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.TFSMLayer(
                        weights_path, call_endpoint="serving_default"
                    )
                ]
            )
            print("[evaluate] Loaded model using TFSMLayer (inference only).")
            return model
        except Exception as e:
            print(f"[evaluate] SavedModel load failed: {e}")

    raise ValueError(f"Unable to load model from path: {weights_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--weights", type=str, default=Config.best_model_path)
    ap.add_argument("--img_size", type=int, default=Config.img_size)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    args = ap.parse_args()

    # Load test dataset
    test_dir = os.path.join(args.data_dir, "test")
    test_ds = make_dataset(test_dir, args.img_size, args.batch_size, Config.seed)

    # Load label map
    with open(Config.label_map_path) as f:
        label_map = json.load(f)
    idx_to_class = [
        (
            label_map[str(i)]
            if isinstance(label_map, dict) and str(i) in label_map
            else label_map[i]
        )
        for i in range(len(test_ds.class_names))
    ]

    # Load model automatically
    model = load_model_auto(args.weights)

    # Predict
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    preds = model.predict(test_ds, verbose=0)
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(preds, axis=1)

    # Report
    print("[evaluate] Classification Report:")
    print(
        classification_report(
            y_true_idx, y_pred_idx, target_names=test_ds.class_names, digits=4
        )
    )

    # Confusion Matrices
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plot_confusion_matrix(
        cm,
        classes=test_ds.class_names,
        normalize=False,
        out_path="reports/confusion_matrix.png",
    )
    plot_confusion_matrix(
        cm,
        classes=test_ds.class_names,
        normalize=True,
        out_path="reports/confusion_matrix_norm.png",
    )


if __name__ == "__main__":
    main()
