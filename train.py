import argparse

from configs.default import TrainConfig
from src.utils import set_seed, get_device
from src.dataset import create_dataloaders
from src.models import create_model
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stanford Dogs breed classifier")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "swin_tiny", "convnext_tiny"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--resume", action="store_true", help="Resume training from best checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(model_name=args.model)

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    device = get_device()

    print(f"Config: {cfg.model_name} ({cfg.timm_model_name})")
    print(f"Device: {device} | Epochs: {cfg.epochs} | BS: {cfg.batch_size} | LR: {cfg.lr}")

    train_loader, val_loader, _ = create_dataloaders(cfg)
    model = create_model(cfg)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    trainer = Trainer(model, cfg, train_loader, val_loader, device)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
