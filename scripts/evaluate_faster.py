"""
Evaluate a trained checkpoint against a test data directory (batched inference).

Batches all mel spectrograms and runs the Whisper encoder in chunks instead of
one file at a time — significantly faster, especially on GPU.

Supports pre-computed embedding caches (from precompute_embeddings.py) to skip
the Whisper encoder entirely — near-instant evaluation after the one-time cache
generation step.

Test directory must mirror training format: test_data/<label>/*.wav

Usage:
    python scripts/evaluate_faster.py --checkpoint models/voice_commander.pth
    python scripts/evaluate_faster.py --checkpoint models/2026-03-30_strategy3_initialTest.pth --test-dir test_data_3cmd --batch-size 64
    python scripts/evaluate_faster.py --checkpoint models/voice_commander.pth --batch-size 64

    # With pre-computed embeddings (skips Whisper encoder):
    python scripts/precompute_embeddings.py --checkpoint models/my_model.pth --test-dir test_data_3cmd
    python scripts/evaluate_faster.py --checkpoint models/2026-03-30_strategy3_initialTest.pth --embeddings test_data_3cmd_embeddings.pt
"""

import argparse
import csv
import logging
import sys
import time
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
        description="Evaluate a voice command checkpoint against a test directory (batched).",
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
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of audio files to process per forward pass",
    )
    parser.add_argument(
        "--embeddings", default=None,
        help="Path to a pre-computed embeddings .pt file (from precompute_embeddings.py). "
             "Skips audio loading and the Whisper encoder entirely.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)
    if not args.embeddings and not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading checkpoint: %s", checkpoint_path.name)
    t0 = time.time()
    model, label_to_idx, idx_to_label, _, _ = load_checkpoint(str(checkpoint_path), device=str(device))
    model.eval()
    n_mels = model.n_mels
    logger.info("Checkpoint loaded in %.1fs  (%d classes)", time.time() - t0, len(label_to_idx))

    # --- Cached-embeddings path (skips audio loading + Whisper encoder) ---
    if args.embeddings:
        embeddings_path = Path(args.embeddings)
        if not embeddings_path.exists():
            logger.error("Embeddings file not found: %s", embeddings_path)
            sys.exit(1)

        logger.info("Loading pre-computed embeddings from %s", embeddings_path)
        t0 = time.time()
        cache = torch.load(embeddings_path, map_location=device)

        # Validate whisper model compatibility
        cached_whisper = cache.get("whisper_model_name", "<unknown>")
        ckpt_raw = torch.load(str(checkpoint_path), map_location="cpu")
        ckpt_whisper = ckpt_raw.get("whisper_model_name", "<unknown>")
        if cached_whisper != ckpt_whisper:
            logger.warning(
                "Whisper model mismatch — cache was built with '%s', checkpoint uses '%s'. "
                "Results may be incorrect. Re-run precompute_embeddings.py with this checkpoint.",
                cached_whisper, ckpt_whisper,
            )

        embeddings: torch.Tensor = cache["embeddings"].to(device)  # [N, hidden_dim]
        all_actual_labels: list[str] = cache["labels"]
        all_filenames: list[str] = cache["filenames"]
        logger.info("Loaded %d embeddings in %.2fs", len(all_actual_labels), time.time() - t0)

        # Validate label set
        checkpoint_labels = set(label_to_idx.keys())
        cache_labels = set(all_actual_labels)
        if checkpoint_labels != cache_labels:
            only_ckpt = checkpoint_labels - cache_labels
            only_cache = cache_labels - checkpoint_labels
            if only_ckpt:
                logger.error("Labels in checkpoint but missing from cache: %s", sorted(only_ckpt))
            if only_cache:
                logger.error("Labels in cache but missing from checkpoint: %s", sorted(only_cache))
            sys.exit(1)

        # Classifier-only inference (one tiny linear layer — near-instant)
        logger.info("Running classifier on %d cached embeddings...", len(embeddings))
        t0 = time.time()
        all_pred_labels: list[str] = []
        all_confidences: list[float] = []
        with torch.no_grad():
            logits = model.classifier(embeddings)  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_idxs = probs.max(dim=-1)
        for pred_idx, conf in zip(pred_idxs.tolist(), confidences.tolist()):
            all_pred_labels.append(idx_to_label[pred_idx])
            all_confidences.append(conf)
        logger.info("Classifier inference done in %.3fs", time.time() - t0)

        all_wav_paths = [Path(fn) for fn in all_filenames]

    else:
        # --- Standard path: load wav files → mel → encoder → classifier ---
        test_data = _scan_test_dir(test_dir)
        if not test_data:
            logger.error("No .wav files found under %s", test_dir)
            sys.exit(1)

        total_wav_count = sum(len(v) for v in test_data.values())
        logger.info(
            "Found %d files across %d classes in %s",
            total_wav_count, len(test_data), test_dir,
        )

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

        # Preprocess all files upfront
        logger.info("Preprocessing %d audio files...", total_wav_count)
        t0 = time.time()
        all_wav_paths: list[Path] = []
        all_actual_labels: list[str] = []
        all_mels: list[torch.Tensor] = []
        all_n_frames: list[int] = []

        processed = 0
        for actual_label, wav_paths in sorted(test_data.items()):
            for wav_path in wav_paths:
                try:
                    mel, n_frames = preprocess_audio(str(wav_path), n_mels=n_mels)
                except Exception as e:
                    logger.warning("Skipping %s — preprocessing failed: %s", wav_path.name, e)
                    continue
                all_wav_paths.append(wav_path)
                all_actual_labels.append(actual_label)
                all_mels.append(mel)
                all_n_frames.append(n_frames)
                processed += 1
                if processed % 50 == 0:
                    logger.info("  Preprocessed %d / %d files...", processed, total_wav_count)

        logger.info("Preprocessing done: %d files in %.1fs", processed, time.time() - t0)

        # Batched inference
        n_batches = (len(all_mels) + args.batch_size - 1) // args.batch_size
        logger.info(
            "Running inference: %d files in %d batch(es) of up to %d",
            len(all_mels), n_batches, args.batch_size,
        )
        t0 = time.time()
        all_pred_labels: list[str] = []
        all_confidences: list[float] = []

        for batch_num, batch_start in enumerate(range(0, len(all_mels), args.batch_size), start=1):
            batch_mels = torch.stack(all_mels[batch_start : batch_start + args.batch_size]).to(device)
            batch_frames = torch.tensor(all_n_frames[batch_start : batch_start + args.batch_size], device=device)
            logger.info("  Batch %d / %d (%d files)...", batch_num, n_batches, len(batch_mels))

            with torch.no_grad():
                logits = model(batch_mels, batch_frames)
                probs = torch.softmax(logits, dim=-1)
                confidences, pred_idxs = probs.max(dim=-1)

            for pred_idx, conf in zip(pred_idxs.tolist(), confidences.tolist()):
                all_pred_labels.append(idx_to_label[pred_idx])
                all_confidences.append(conf)

        logger.info("Inference done in %.1fs", time.time() - t0)

    # CSV output path
    csv_dir = Path("model_eval")
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / f"{checkpoint_path.stem}_evaluation.csv"

    rows = []
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    print()
    print(f"{'FILE':<45} {'ACTUAL':<25} {'PREDICTED':<25} {'CONF':>6}  {'':>6}")
    print("-" * 115)

    for wav_path, actual_label, pred_label, conf in zip(
        all_wav_paths, all_actual_labels, all_pred_labels, all_confidences
    ):
        is_correct = pred_label == actual_label
        indicator = "[PASS]" if is_correct else "[FAIL]"
        print(f"{wav_path.name:<45} {actual_label:<25} {pred_label:<25} {conf:>6.1%}  {indicator}")

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
