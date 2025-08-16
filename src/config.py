from dataclasses import dataclass


@dataclass
class Config:
    img_size: int = 224
    channels: int = 3
    seed: int = 42
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    val_split: float = 0.15
    test_split: float = 0.15
    augment: bool = True
    label_map_path: str = "artifacts/label_map.json"
    best_model_path = "artifacts/best_model.keras"
    history_path: str = "artifacts/history.json"
