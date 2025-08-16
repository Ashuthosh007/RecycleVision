import argparse, os, json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras import callbacks, optimizers
from config import Config
from models import build_model, unfreeze_top_layers


def build_datasets(
    data_dir: str, img_size: int, batch_size: int, seed: int, augment: bool = True
):
    autotune = tf.data.AUTOTUNE
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    AUG = (
        tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.05, 0.05),
            ]
        )
        if augment
        else tf.keras.Sequential([])
    )

    def aug_fn(x, y):
        return AUG(x, training=True), y

    if augment:
        train_ds = train_ds.map(aug_fn, num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "resnet50", "efficientnetb0"],
    )
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--img_size", type=int, default=Config.img_size)
    ap.add_argument(
        "--fine_tune",
        action="store_true",
        help="Unfreeze top layers for fine-tuning after warmup",
    )
    ap.add_argument("--save_format", type=str, default="keras", choices=["keras", "h5"])
    args = ap.parse_args()

    train_ds, val_ds, class_names = build_datasets(
        args.data_dir,
        args.img_size,
        args.batch_size,
        Config.seed,
        augment=Config.augment,
    )

    # Save label map
    os.makedirs("artifacts", exist_ok=True)
    with open(Config.label_map_path, "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

    # Compute class weights for imbalance handling
    y_train_indices = []
    for _, y in train_ds.unbatch():
        y_train_indices.append(np.argmax(y.numpy()))
    y_train_indices = np.array(y_train_indices)

    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=y_train_indices
    )
    class_weights = dict(enumerate(class_weights_array))
    print("[train] Computed class weights:", class_weights)

    # Define checkpoint filename format (decides save format)
    ckpt_path = (
        Config.best_model_path
        if args.save_format == "keras"
        else Config.best_model_path.replace(".keras", ".h5")
    )

    # Load existing model if available
    if os.path.exists(ckpt_path):
        print(f"[train] Found existing model at {ckpt_path}, loading...")
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        model.compile(
            optimizer=optimizers.Adam(Config.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        print("[train] No existing model found. Building new model...")
        model = build_model(
            args.model,
            (args.img_size, args.img_size, Config.channels),
            len(class_names),
        )
        model.compile(
            optimizer=optimizers.Adam(Config.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Checkpoints (format is determined by ckpt_path extension)
    ckpt = callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    es = callbacks.EarlyStopping(
        patience=4, restore_best_weights=True, monitor="val_accuracy", mode="max"
    )

    # Warmup training
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(1, args.epochs // 2),
        callbacks=[ckpt, es],
        class_weight=class_weights,
    )

    # Fine-tuning
    if args.fine_tune:
        model = unfreeze_top_layers(model, pct=0.2)
        model.compile(
            optimizer=optimizers.Adam(Config.lr * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs - len(hist.history["loss"]),
            callbacks=[ckpt, es],
            class_weight=class_weights,
        )
        for k, v in hist2.history.items():
            hist.history[k] = hist.history.get(k, []) + v

    # Save history
    with open(Config.history_path, "w") as f:
        json.dump(hist.history, f, indent=2)

    print("[train] Done. Best model saved to", ckpt_path)


if __name__ == "__main__":
    main()
