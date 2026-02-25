"""
Usage:
    python scripts/train.py
    python scripts/train.py --data-dir data/ --whisper-model small --epochs 20
    python scripts/train.py --unfreeze-encoder
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from model.classifier import WhisperCommandClassifier
from training.dataset import build_dataloaders, load_data_from_dir
from training.trainer import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data"
DEFAULT_CHECKPOINT = "models/voice_commander.pth"
DEFAULT_WHISPER_MODEL = "turbo"
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper voice command classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Root data directory (one subfolder per command class)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Where to save the best model checkpoint")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "turbo"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Fine-tune the full Whisper encoder (slower, needs more data)")
    args = parser.parse_args()

    logger.info(f"Loading data from '{args.data_dir}'...")
    data_dict = load_data_from_dir(args.data_dir)
    if not data_dict:
        logger.error(f"No .wav files found in '{args.data_dir}'")
        sys.exit(1)

    unique_labels = sorted(data_dict.keys())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    logger.info(f"Classes ({len(unique_labels)}): {unique_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    freeze = not args.unfreeze_encoder
    model = WhisperCommandClassifier(args.whisper_model, len(unique_labels), freeze)
    model.to(device)

    train_loader, val_loader = build_dataloaders(data_dict, label_to_idx, model.n_mels, args.batch_size)
    logger.info(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    train_model(
        model, train_loader, val_loader, device,
        args.epochs, args.lr, args.checkpoint,
        label_to_idx, idx_to_label, args.whisper_model, freeze,
    )

if __name__ == "__main__":
    main()