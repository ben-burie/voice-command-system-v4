"""
Evaluate a trained checkpoint against a test data directory.

Test directory must mirror training format: test_data/<label>/*.wav

Usage:
    python scripts/evaluate.py --checkpoint models/voice_commander.pth
    python scripts/evaluate.py --checkpoint models/2026-03-22_simple_model_for_continual.pth --test-dir test_data_3cmd
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from audio.preprocessing import preprocess_audio
from model.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _scan_test_dir(test_dir: Path) -> dict[str, list[Path]]:
    """Return {label: [wav_path, ...]} for all subdirectories containing .wav files."""
    result = {}
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        wavs = sorted(subdir.glob("*.wav"))
        if wavs:
            result[subdir.name] = wavs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a voice command checkpoint against a test directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pth model checkpoint to evaluate",
    )
    parser.add_argument(
        "--test-dir", default="test_data",
        help="Root test directory (one subfolder per command label)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, label_to_idx, idx_to_label, _, _ = load_checkpoint(str(checkpoint_path), device=str(device))
    model.eval()
    n_mels = model.n_mels

    test_data = _scan_test_dir(test_dir)
    if not test_data:
        logger.error("No .wav files found under %s", test_dir)
        sys.exit(1)

    # Strict label match — fail loudly on any mismatch
    checkpoint_labels = set(label_to_idx.keys())
    test_labels = set(test_data.keys())
    if checkpoint_labels != test_labels:
        only_ckpt = checkpoint_labels - test_labels
        only_test = test_labels - checkpoint_labels
        if only_ckpt:
            logger.error("Labels in checkpoint but missing from test_data: %s", sorted(only_ckpt))
        if only_test:
            logger.error("Labels in test_data but missing from checkpoint: %s", sorted(only_test))
        sys.exit(1)

    # CSV output path: model_eval/ directory, stem + "_evaluation.csv"
    csv_dir = Path("model_eval")
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / f"{checkpoint_path.stem}_evaluation.csv"

    rows = []  # accumulate for CSV

    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    print()
    print(f"{'FILE':<45} {'ACTUAL':<25} {'PREDICTED':<25} {'CONF':>6}  {'':>6}")
    print("-" * 115)

    for actual_label, wav_paths in sorted(test_data.items()):
        for wav_path in wav_paths:
            try:
                mel, n_frames = preprocess_audio(str(wav_path), n_mels=n_mels)
            except Exception as e:
                logger.warning("Skipping %s — preprocessing failed: %s", wav_path.name, e)
                continue

            mel = mel.unsqueeze(0).to(device)
            n_frames_t = torch.tensor([n_frames], device=device)

            with torch.no_grad():
                logits = model(mel, n_frames_t)
                probs = torch.softmax(logits, dim=-1)
                confidence, pred_idx = probs.max(dim=-1)

            pred_label = idx_to_label[pred_idx.item()]
            conf = confidence.item()
            is_correct = pred_label == actual_label

            indicator = "[PASS]" if is_correct else "[FAIL]"
            print(
                f"{wav_path.name:<45} {actual_label:<25} {pred_label:<25} {conf:>6.1%}  {indicator}"
            )

            rows.append({
                "file": wav_path.name,
                "actual_label": actual_label,
                "predicted_label": pred_label,
                "confidence": f"{conf:.4f}",
                "correct": is_correct,
            })

            per_class_total[actual_label] += 1
            if is_correct:
                per_class_correct[actual_label] += 1
                correct += 1
            total += 1

    # Summary
    overall_acc = correct / total if total else 0.0
    print()
    print("=" * 60)
    print(f"OVERALL ACCURACY: {correct}/{total}  ({overall_acc:.1%})")
    print()
    print(f"{'LABEL':<30} {'CORRECT':>8} {'TOTAL':>8} {'ACCURACY':>10}")
    print("-" * 60)
    for label in sorted(per_class_total):
        n = per_class_total[label]
        c = per_class_correct[label]
        print(f"{label:<30} {c:>8} {n:>8} {c/n:>10.1%}")
    print("=" * 60)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "actual_label", "predicted_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
