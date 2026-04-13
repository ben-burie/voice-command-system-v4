"""
Continual learning: add one new command to an existing checkpoint without
disturbing previously learned classes.

Flow:
  1. Load existing checkpoint
  2. Prompt for new command details → update commands.yaml
  3. Generate synthetic data for the new command via generate_data.py
  4. Build an expanded (N+1)-class model, copying old head weights into rows 0..N-1
  5. Train using the selected strategy (see below)
  6. Save a new checkpoint with updated label maps

Training strategies (--strategy):
  1  New-only, hard freeze  — train only on new command data; old head rows are
                              always gradient-zeroed so they never change.
  2  New-only, soft freeze  — train only on new command data; per batch, old head
                              rows update with probability --grad-update-prob (0.0 =
                              always zero, 1.0 = always update). Allows controlled
                              plasticity at the cost of some forgetting risk.
  3  Replay (percentage)    — train on all new command data plus --replay-pct of
                              each old class's available .wav files (e.g. 0.5 = 50%).
                              Old head rows are still gradient-zeroed; only the new
                              row is updated. Replay data keeps old classes visible
                              to the loss without touching their weights.

Usage:
    python scripts/continual_train.py --checkpoint models/old.pth --strategy 1
    python scripts/continual_train.py --checkpoint models/old.pth --strategy 2 --grad-update-prob 0.3
    python scripts/continual_train.py --checkpoint models/old.pth --strategy 3 --replay-pct 0.5
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

def prompt_new_command(
    label: str | None = None,
    display_name: str | None = None,
    action: str | None = None,
    url: str | None = None,
    browser: str | None = None,
    app: str | None = None,
) -> tuple[str, dict]:
    """Prompt for new command metadata. Returns (label, commands.yaml entry dict).

    Any argument supplied via CLI skips the corresponding interactive prompt.
    """
    print("\n=== New Command Setup ===")

    if label is None:
        label = input("Label key (e.g. Open_Spotify): ").strip()
        while not label:
            label = input("Label cannot be empty. Label key: ").strip()

    default_display = label.replace("_", " ")
    if display_name is None:
        display_name = input(f"Display name [{default_display}]: ").strip() or default_display

    if action is None:
        print(f"\nAvailable actions: {', '.join(AVAILABLE_ACTIONS)}")
        action = input("Action: ").strip()
        while action not in AVAILABLE_ACTIONS:
            action = input(f"Must be one of {AVAILABLE_ACTIONS}: ").strip()

    params: dict = {}
    if action == "open_url":
        params["url"] = url or input("URL: ").strip()
    elif action == "open_url_in_browser":
        params["url"] = url or input("URL: ").strip()
        params["browser"] = browser or input("Browser (e.g. brave, chrome): ").strip()
    elif action == "open_application":
        if app is None:
            app = input("App name (leave blank for none): ").strip() or None
        params["app"] = app

    return label, {"display_name": display_name, "action": action, "params": params}


def prompt_training_config(model_name: str | None = None, epochs: int | None = None) -> tuple[str, int]:
    """Prompt for output model name and epoch count (mirrors train.py style).

    Arguments supplied via CLI skip the corresponding interactive prompt.
    """
    if model_name is None:
        model_name = input("\nEnter name for the new checkpoint: ").strip()
    if epochs is None:
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
    replay_pct: float = 0.0,
) -> tuple[list[str], list[str]]:
    """
    Returns (file_paths, labels) for:
      - All .wav files in data_dir/new_label
      - replay_pct fraction of .wav files per old class (strategy 3); 0.0 = no replay
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

    if replay_pct > 0.0:
        for old_label in old_labels:
            old_dir = data_dir / old_label
            if not old_dir.exists():
                logger.warning("Replay: data dir not found for '%s', skipping.", old_label)
                continue
            old_files = [str(f) for f in old_dir.iterdir() if f.suffix.lower() == ".wav"]
            n_replay = max(1, round(len(old_files) * replay_pct))
            sampled = random.sample(old_files, min(n_replay, len(old_files)))
            file_paths.extend(sampled)
            labels.extend([old_label] * len(sampled))
            logger.info("Replay '%s': %d / %d files (%.0f%%)",
                        old_label, len(sampled), len(old_files), replay_pct * 100)

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
    grad_update_prob: float = 0.0,
    freeze_encoder: bool = True,
) -> None:
    """
    Standard train/val loop.

    grad_update_prob controls how often old head rows are allowed to update:
      0.0 — always zero old rows (strategies 1 and 3, hard freeze)
      1.0 — never zero old rows (full plasticity)
      0 < p < 1 — per batch, zero old rows with probability (1 - p) (strategy 2)
    """
    # Only pass the head parameters — encoder is already frozen
    optimizer = torch.optim.AdamW(
        [model.classifier.weight, model.classifier.bias], lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train() # PyTorch module that sets model into training mode (activates dropout and batchnorm) - since the model uses none of these this line has no effect
        t_loss = t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            # Move data from CPU to GPU since the model runs on GPU, these need to be there too to do computations
            mels = mels.to(device) # moves mels tensor from CPU to GPU for colab
            label_idxs = label_idxs.to(device) # moves label_idx from CPU to GPU for colab
            n_frames = n_frames.to(device) # moves n_frames to GPU for colab

            optimizer.zero_grad() # clear gradients from previous batch
            logits = model(mels, n_frames) # makes a prediction (logit = numerical assignment of each output command)
            loss = criterion(logits, label_idxs) # Calculate loss
            loss.backward() # Compute the gradient

            # Per batch, decide whether to zero old head row gradients.
            # If random() >= grad_update_prob the old rows are frozen this step.
            if random.random() >= grad_update_prob:
                with torch.no_grad():
                    if model.classifier.weight.grad is not None:
                        model.classifier.weight.grad[:n_old] = 0
                    if model.classifier.bias.grad is not None:
                        model.classifier.bias.grad[:n_old] = 0

            optimizer.step() # Adjust weights

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
                whisper_model_name, freeze_encoder, v_acc, epoch + 1,
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
    parser.add_argument("--n-samples", type=int, default=150,
                        help="Target .wav files to generate for the new command")
    parser.add_argument("--strategy", type=int, default=1, choices=[1, 2, 3],
                        help="Training strategy: 1=new-only hard freeze, "
                             "2=new-only soft freeze (stochastic grad update), "
                             "3=new + percentage replay of old classes. "
                             "Strategies 2 and 3 can be combined by passing both "
                             "--grad-update-prob >0 and --replay-pct >0 with --strategy 3.")
    parser.add_argument("--grad-update-prob", type=float, default=0.0,
                        help="Probability per batch that old head rows are allowed to update "
                             "(0.0=always frozen, 1.0=always update). Active for strategy 2; "
                             "also accepted by strategy 3 to enable combined replay+soft-freeze.")
    parser.add_argument("--replay-pct", type=float, default=0.5,
                        help="[Strategy 3] Fraction of each old class's .wav files to "
                             "include as replay data (e.g. 0.5 = 50%%)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Non-interactive overrides — supply these to skip all prompts (useful for Colab scripting)
    parser.add_argument("--model-name", default=None,
                        help="Output checkpoint name, e.g. 'run1_s2_p03' (skips interactive prompt)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (skips interactive prompt)")
    parser.add_argument("--new-label", default=None,
                        help="Label key for the new command, e.g. 'Open_Spotify' (skips prompt)")
    parser.add_argument("--new-display-name", default=None,
                        help="Display name for the new command (skips prompt; defaults to label with underscores replaced)")
    parser.add_argument("--new-action", default=None, choices=AVAILABLE_ACTIONS,
                        help="Action type for the new command (skips prompt)")
    parser.add_argument("--new-url", default=None,
                        help="URL param for open_url / open_url_in_browser actions (skips prompt)")
    parser.add_argument("--new-browser", default=None,
                        help="Browser param for open_url_in_browser action (skips prompt)")
    parser.add_argument("--new-app", default=None,
                        help="App param for open_application action (skips prompt)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip synthetic data generation (data already exists in --data-dir)")
    args = parser.parse_args()

    if args.strategy in (2, 3) and not (0.0 <= args.grad_update_prob <= 1.0):
        parser.error("--grad-update-prob must be between 0.0 and 1.0")
    if args.strategy == 3 and not (0.0 < args.replay_pct <= 1.0):
        parser.error("--replay-pct must be between 0.0 (exclusive) and 1.0")

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load existing checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    old_model, label_to_idx, idx_to_label, whisper_model_name, base_freeze_encoder = load_checkpoint(
        args.checkpoint, str(device)
    )
    n_old = len(label_to_idx)
    old_labels = list(label_to_idx.keys())
    logger.info("Existing classes (%d): %s", n_old, old_labels)

    # 2. Prompt for new command details + training config (CLI args skip the prompts)
    new_label, entry = prompt_new_command(
        label=args.new_label,
        display_name=args.new_display_name,
        action=args.new_action,
        url=args.new_url,
        browser=args.new_browser,
        app=args.new_app,
    )
    if new_label in label_to_idx:
        logger.error("Label '%s' already exists in this checkpoint. Aborting.", new_label)
        sys.exit(1)

    model_name, epochs = prompt_training_config(args.model_name, args.epochs)
    checkpoint_out = f"models/{date.today()}_{model_name}.pth"

    # 3. Update commands.yaml
    update_commands_yaml(config_path, new_label, entry)

    # 4. Generate synthetic data for the new command
    if args.skip_generation:
        logger.info("Skipping data generation (--skip-generation set).")
    else:
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

    # 7. Collect training data according to chosen strategy
    if args.strategy == 3:
        replay_pct = args.replay_pct
        grad_update_prob = args.grad_update_prob  # 0.0 = hard freeze (default); >0 = combined
        if grad_update_prob > 0.0:
            logger.info(
                "Strategy 3+2 (combined): %.0f%% replay per old class, old rows update with p=%.2f",
                replay_pct * 100, grad_update_prob,
            )
        else:
            logger.info("Strategy 3: %.0f%% replay per old class, old rows always frozen", replay_pct * 100)
    elif args.strategy == 2:
        replay_pct = 0.0
        grad_update_prob = args.grad_update_prob
        logger.info("Strategy 2: new-only data, old rows update with p=%.2f per batch", grad_update_prob)
    else:
        replay_pct = 0.0
        grad_update_prob = 0.0
        logger.info("Strategy 1: new-only data, old rows always frozen")

    file_paths, labels = collect_files(new_label, data_dir, old_labels, replay_pct=replay_pct)
    train_loader, val_loader = build_dataloaders(
        file_paths, labels, label_to_idx, model.n_mels, args.batch_size
    )
    logger.info("Train: %d  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # 8. Train
    logger.info("Training new class '%s' for %d epochs → %s", new_label, epochs, checkpoint_out)
    train_continual(
        model, train_loader, val_loader, device,
        epochs, args.lr, n_old,
        checkpoint_out, label_to_idx, idx_to_label, whisper_model_name,
        grad_update_prob=grad_update_prob,
        freeze_encoder=base_freeze_encoder,
    )


if __name__ == "__main__":
    main()
