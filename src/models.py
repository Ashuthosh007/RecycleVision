
from typing import Tuple
import tensorflow as tf

def _top(class_count: int) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(class_count, activation="softmax")
    ])

def build_backbone(name: str, input_shape: Tuple[int,int,int]):
    name = name.lower()
    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        preprocess = tf.keras.applications.resnet.preprocess_input
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from mobilenetv2, resnet50, efficientnetb0.")
    return base, preprocess

def build_model(name: str, input_shape: Tuple[int,int,int], num_classes: int) -> tf.keras.Model:
    base, preprocess = build_backbone(name, input_shape)
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    outputs = _top(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def unfreeze_top_layers(model: tf.keras.Model, pct: float = 0.2):
    # unfreeze top pct of base layers for fine-tuning
    base = None
    for layer in model.layers:
        if hasattr(layer, "name") and isinstance(layer, tf.keras.Model) and layer.name not in ["sequential","functional"]:
            base = layer
    if base is None:
        # fallback: assume second layer is base
        base = model.layers[2]
    total = len(base.layers)
    cutoff = int(total * (1 - pct))
    for i, l in enumerate(base.layers):
        l.trainable = (i >= cutoff)
    return model
