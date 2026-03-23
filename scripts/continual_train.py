"""
Continual learning: add one new command to an existing checkpoint without
disturbing previously learned classes.

Flow:
  1. Load existing checkpoint
  2. Prompt for new command details → update commands.yaml
  3. Generate synthetic data for the new command via generate_data.py
  4. Build an expanded (N+1)-class model, copying old head weights into rows 0..N-1
  5. Train with gradient zeroing on old rows — only the new row updates
  6. Save a new checkpoint with updated label maps

Usage:
    python scripts/continual_train.py --checkpoint models/old.pth
    python scripts/continual_train.py --checkpoint models/old.pth --replay-samples 50 --epochs 20
"""

import argparse
import logging
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.checkpoint import load_checkpoint, save_checkpoint
from model.classifier import WhisperCommandClassifier
from training.dataset import CommandDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AVAILABLE_ACTIONS = ["open_url", "open_url_in_browser", "open_application"]


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_new_command() -> tuple[str, dict]:
    """Prompt for new command metadata. Returns (label, commands.yaml entry dict)."""
    print("\n=== New Command Setup ===")

    label = input("Label key (e.g. Open_Spotify): ").strip()
    while not label:
        label = input("Label cannot be empty. Label key: ").strip()

    default_display = label.replace("_", " ")
    display_name = input(f"Display name [{default_display}]: ").strip() or default_display

    print(f"\nAvailable actions: {', '.join(AVAILABLE_ACTIONS)}")
    action = input("Action: ").strip()
    while action not in AVAILABLE_ACTIONS:
        action = input(f"Must be one of {AVAILABLE_ACTIONS}: ").strip()

    params: dict = {}
    if action == "open_url":
        params["url"] = input("URL: ").strip()
    elif action == "open_url_in_browser":
        params["url"] = input("URL: ").strip()
        params["browser"] = input("Browser (e.g. brave, chrome): ").strip()
    elif action == "open_application":
        app = input("App name (leave blank for none): ").strip()
        params["app"] = app or None

    return label, {"display_name": display_name, "action": action, "params": params}


def prompt_training_config() -> tuple[str, int]:
    """Prompt for output model name and epoch count (mirrors train.py style)."""
    model_name = input("\nEnter name for the new checkpoint: ").strip()
    epochs = int(input("Number of epochs: "))
    return model_name, epochs


# ---------------------------------------------------------------------------
# commands.yaml
# ---------------------------------------------------------------------------

def update_commands_yaml(config_path: Path, label: str, entry: dict) -> None:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    data.setdefault("commands", {})[label] = entry
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Updated %s with label '%s'", config_path, label)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def run_data_generation(new_label: str, n_samples: int, data_dir: Path, config_path: Path) -> None:
    script = Path(__file__).parent / "generate_data.py"
    cmd = [
        sys.executable, str(script),
        "--commands", new_label,
        "--n-samples", str(n_samples),
        "--data-dir", str(data_dir),
        "--config", str(config_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Data generation failed (exit %d).", result.returncode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model expansion
# ---------------------------------------------------------------------------

def expand_classifier(
    old_model: WhisperCommandClassifier,
    whisper_model_name: str,
    n_total: int,
    device: torch.device,
) -> WhisperCommandClassifier:
    """
    Build a new n_total-class model.
    - Encoder weights copied from old_model (stays frozen).
    - Head rows 0..n_old-1 copied from old head.
    - New row (n_old) is randomly initialized by default.
    """
    new_model = WhisperCommandClassifier(whisper_model_name, n_total, freeze_encoder=True)
    new_model.encoder.load_state_dict(old_model.encoder.state_dict())

    n_old = old_model.classifier.out_features
    with torch.no_grad():
        new_model.classifier.weight[:n_old].copy_(old_model.classifier.weight)
        new_model.classifier.bias[:n_old].copy_(old_model.classifier.bias)

    new_model.to(device)
    return new_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_files(
    new_label: str,
    data_dir: Path,
    old_labels: list[str],
    replay_samples: int,
) -> tuple[list[str], list[str]]:
    """
    Returns (file_paths, labels) for:
      - All .wav files in data_dir/new_label
      - Up to replay_samples random files per old class
    """
    file_paths: list[str] = []
    labels: list[str] = []

    new_dir = data_dir / new_label
    if not new_dir.exists():
        logger.error("New command data dir not found: %s", new_dir)
        sys.exit(1)
    new_files = [str(f) for f in new_dir.iterdir() if f.suffix.lower() == ".wav"]
    if not new_files:
        logger.error("No .wav files found in %s", new_dir)
        sys.exit(1)
    file_paths.extend(new_files)
    labels.extend([new_label] * len(new_files))
    logger.info("New class '%s': %d files", new_label, len(new_files))

    if replay_samples > 0:
        for old_label in old_labels:
            old_dir = data_dir / old_label
            if not old_dir.exists():
                logger.warning("Replay: data dir not found for '%s', skipping.", old_label)
                continue
            old_files = [str(f) for f in old_dir.iterdir() if f.suffix.lower() == ".wav"]
            sampled = random.sample(old_files, min(replay_samples, len(old_files)))
            file_paths.extend(sampled)
            labels.extend([old_label] * len(sampled))
            logger.info("Replay '%s': %d files", old_label, len(sampled))

    return file_paths, labels


def build_dataloaders(
    file_paths: list[str],
    labels: list[str],
    label_to_idx: dict,
    n_mels: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ds = CommandDataset(train_paths, train_labels, label_to_idx, n_mels, augment=True)
    val_ds = CommandDataset(val_paths, val_labels, label_to_idx, n_mels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_continual(
    model: WhisperCommandClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    n_old: int,
    checkpoint_path: str,
    label_to_idx: dict,
    idx_to_label: dict,
    whisper_model_name: str,
) -> None:
    """
    Standard train/val loop with one addition: after each backward pass, old
    head rows' gradients are zeroed so only the new class row gets updated.
    """
    # Only pass the head parameters — encoder is already frozen
    optimizer = torch.optim.AdamW(
        [model.classifier.weight, model.classifier.bias], lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            mels = mels.to(device)
            label_idxs = label_idxs.to(device)
            n_frames = n_frames.to(device)

            optimizer.zero_grad()
            logits = model(mels, n_frames)
            loss = criterion(logits, label_idxs)
            loss.backward()

            # Zero gradients for old class rows before the weight update
            with torch.no_grad():
                if model.classifier.weight.grad is not None:
                    model.classifier.weight.grad[:n_old] = 0
                if model.classifier.bias.grad is not None:
                    model.classifier.bias.grad[:n_old] = 0

            optimizer.step()

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total += label_idxs.size(0)

        # --- Validate ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for mels, label_idxs, n_frames in val_loader:
                mels = mels.to(device)
                label_idxs = label_idxs.to(device)
                n_frames = n_frames.to(device)
                logits = model(mels, n_frames)
                loss = criterion(logits, label_idxs)
                v_loss += loss.item()
                v_correct += (logits.argmax(1) == label_idxs).sum().item()
                v_total += label_idxs.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        logger.info(
            "Epoch %02d/%02d | Train %.4f / %.1f%% | Val %.4f / %.1f%%",
            epoch + 1, epochs,
            t_loss / len(train_loader), t_acc,
            v_loss / len(val_loader), v_acc,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, True, v_acc, epoch + 1,
            )
            logger.info("  → Best checkpoint saved (val_acc=%.1f%%)", v_acc)

    logger.info("Training complete. Best val accuracy: %.1f%%", best_val_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual learning: add one new command to an existing checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to existing .pth checkpoint to extend")
    parser.add_argument("--data-dir", default="data",
                        help="Root data directory (one subfolder per command class)")
    parser.add_argument("--config", default="config/commands.yaml",
                        help="Path to commands.yaml")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Target .wav files to generate for the new command")
    parser.add_argument("--replay-samples", type=int, default=0,
                        help="Files per old class to include as replay (0 = no replay)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load existing checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    old_model, label_to_idx, idx_to_label, whisper_model_name, _ = load_checkpoint(
        args.checkpoint, str(device)
    )
    n_old = len(label_to_idx)
    old_labels = list(label_to_idx.keys())
    logger.info("Existing classes (%d): %s", n_old, old_labels)

    # 2. Prompt for new command details + training config
    new_label, entry = prompt_new_command()
    if new_label in label_to_idx:
        logger.error("Label '%s' already exists in this checkpoint. Aborting.", new_label)
        sys.exit(1)

    model_name, epochs = prompt_training_config()
    checkpoint_out = f"models/{date.today()}_{model_name}.pth"

    # 3. Update commands.yaml
    update_commands_yaml(config_path, new_label, entry)

    # 4. Generate synthetic data for the new command
    run_data_generation(new_label, args.n_samples, data_dir, config_path)

    # 5. Expand label maps (new label appended at index N, old indices unchanged)
    new_idx = n_old
    label_to_idx[new_label] = new_idx
    idx_to_label[new_idx] = new_label
    logger.info("New label '%s' assigned index %d", new_label, new_idx)

    # 6. Build expanded model (free old model from memory first)
    logger.info("Building %d-class model (was %d)...", n_old + 1, n_old)
    model = expand_classifier(old_model, whisper_model_name, n_old + 1, device)
    del old_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 7. Collect training data (new command + optional replay)
    file_paths, labels = collect_files(new_label, data_dir, old_labels, args.replay_samples)
    train_loader, val_loader = build_dataloaders(
        file_paths, labels, label_to_idx, model.n_mels, args.batch_size
    )
    logger.info("Train: %d  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # 8. Train — only the new head row accumulates gradient updates
    logger.info("Training new class '%s' for %d epochs → %s", new_label, epochs, checkpoint_out)
    train_continual(
        model, train_loader, val_loader, device,
        epochs, args.lr, n_old,
        checkpoint_out, label_to_idx, idx_to_label, whisper_model_name,
    )


if __name__ == "__main__":
    main()
