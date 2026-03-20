from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Model
    model_name: str = "vit_small"
    num_classes: int = 120
    pretrained: bool = False

    # Data
    data_dir: Path = Path("data")
    image_size: int = 224
    val_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.05

    # Scheduler
    scheduler: str = "cosine"

    # Training
    epochs: int = 50
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10

    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    # Misc
    seed: int = 42
    mixed_precision: bool = True

    # timm model name mapping
    MODEL_REGISTRY: dict = field(default_factory=lambda: {
        "vit_small": "vit_small_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "convnext_tiny": "convnext_tiny",
    })

    @property
    def timm_model_name(self) -> str:
        return self.MODEL_REGISTRY[self.model_name]
