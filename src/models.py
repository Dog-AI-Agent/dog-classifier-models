import timm
import torch.nn as nn

from configs.default import TrainConfig


def create_model(cfg: TrainConfig) -> nn.Module:
    """Create a timm model from config."""
    model = timm.create_model(
        cfg.timm_model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )
    return model
