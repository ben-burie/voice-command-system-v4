import csv
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_CHECKPOINT = "models/2026-03-22_simple_model_for_continual.pth"
TEST_DIR_BASE = "test_data_original_3"   # 3-command test set (base model evaluation)
TEST_DIR_CONTINUAL = "test_data"         # 4-command test set (continual model evaluation)

TRAINING_DATASET_150 = "batch_train_150"
TRAINING_DATASET_300 = "batch_train_300"
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-4

TODAY = date.today().isoformat()

# New command to add via continual training
NEW_LABEL = "Open_Amazon"
NEW_ACTION = "open_url_in_browser"
NEW_URL = "https://amazon.com"
NEW_BROWSER = "brave"
NEW_DATA_150 = 150   # --n-samples for the new command (first batch of runs)
NEW_DATA_300 = 300   # --n-samples for the new command (second batch of runs)

RESULTS_CSV = f"model_eval/batch_results_{TODAY}.csv"

# ── 12 test configurations from the test table (2026-04-13) ──────────────────
#
#  Test  | Grad update prob | Replay pct | Model name
#  ------+------------------+------------+-----------
#   1    |       0.0        |    0.0     | s1_base       (strategy 1 baseline)
#   2    |       0.3        |    0.0     | s2_p30        (strategy 2)
#   3    |       0.7        |    0.0     | s2_p70
#   4    |       1.0        |    0.0     | s2_p100
#   5    |       0.0        |    0.1     | s3_r10        (strategy 3 replay-only)
#   6    |       0.0        |    0.3     | s3_r30
#   7    |       0.0        |    0.5     | s3_r50
#   8    |       0.0        |    1.0     | s3_r100
#   9    |       0.3        |    0.3     | s32_p30_r30   (strategy 3+2 combined)
#  10    |       0.3        |    0.7     | s32_p30_r70
#  11    |       0.7        |    0.3     | s32_p70_r30
#  12    |       0.7        |    0.7     | s32_p70_r70

RUNS = [
    {"strategy": 1, "grad-update-prob": 0.0, "replay-pct": 0.0, "model-name": "s1_base"},
    {"strategy": 2, "grad-update-prob": 0.3, "replay-pct": 0.0, "model-name": "s2_p30"},
    {"strategy": 2, "grad-update-prob": 0.7, "replay-pct": 0.0, "model-name": "s2_p70"},
    {"strategy": 2, "grad-update-prob": 1.0, "replay-pct": 0.0, "model-name": "s2_p100"},
    {"strategy": 3, "grad-update-prob": 0.0, "replay-pct": 0.1, "model-name": "s3_r10"},
    {"strategy": 3, "grad-update-prob": 0.0, "replay-pct": 0.3, "model-name": "s3_r30"},
    {"strategy": 3, "grad-update-prob": 0.0, "replay-pct": 0.5, "model-name": "s3_r50"},
    {"strategy": 3, "grad-update-prob": 0.0, "replay-pct": 1.0, "model-name": "s3_r100"},
    {"strategy": 3, "grad-update-prob": 0.3, "replay-pct": 0.3, "model-name": "s32_p30_r30"},
    {"strategy": 3, "grad-update-prob": 0.3, "replay-pct": 0.7, "model-name": "s32_p30_r70"},
    {"strategy": 3, "grad-update-prob": 0.7, "replay-pct": 0.3, "model-name": "s32_p70_r30"},
    {"strategy": 3, "grad-update-prob": 0.7, "replay-pct": 0.7, "model-name": "s32_p70_r70"},
]

# ── Results tracking ──────────────────────────────────────────────────────────

results_rows: list[dict] = []


def load_eval_results(checkpoint_path: str) -> dict:
    """Read the per-model evaluation CSV produced by evaluate.py and return accuracy metrics."""
    eval_csv = Path("model_eval") / f"{Path(checkpoint_path).stem}_evaluation.csv"
    if not eval_csv.exists():
        print(f"[WARNING] Evaluation CSV not found: {eval_csv}")
        return {}

    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)

    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            label = row["actual_label"]
            per_class_total[label] += 1
            if row["correct"] == "True":
                per_class_correct[label] += 1

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())

    metrics = {"overall_accuracy": f"{correct / total:.4f}" if total else "0.0000"}
    for label in sorted(per_class_total):
        n = per_class_total[label]
        c = per_class_correct[label]
        metrics[f"{label}_accuracy"] = f"{c / n:.4f}" if n else "0.0000"

    return metrics


def _flush_results_csv() -> None:
    """Rewrite the master CSV from the current results_rows list."""
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in results_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    Path("model_eval").mkdir(exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"[RESULTS] {RESULTS_CSV} updated ({len(results_rows)} rows)")


def record_result(run_info: dict, checkpoint_path: str) -> None:
    """Append a result row to the in-memory list and rewrite the master CSV."""
    metrics = load_eval_results(checkpoint_path)
    results_rows.append({**run_info, **metrics})
    _flush_results_csv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_evaluate(checkpoint_path: str, test_dir: str) -> None:
    subprocess.run([
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", checkpoint_path,
        "--test-dir", test_dir,
    ], check=True)


def run_base_model(name: str, data_dir: str) -> str:
    """Train a base model, evaluate it, record results. Returns the checkpoint path."""
    checkpoint_path = f"models/{TODAY}_{name}.pth"

    print(f"\n{'='*60}")
    print(f"Training base model: {name}")
    print(f"{'='*60}")
    t_start = time.time()
    subprocess.run([
        sys.executable, "scripts/train.py",
        "--data-dir", data_dir,
        "--whisper-model", "turbo",
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--model-name", name,
        "--epochs", str(EPOCHS),
    ], check=True)
    elapsed = time.time() - t_start
    print(f"[TIMING] {name}: {elapsed:.1f}s")

    run_evaluate(checkpoint_path, TEST_DIR_BASE)
    record_result(
        {
            "model_name": name,
            "type": "base",
            "base_model": "",
            "strategy": "",
            "grad_update_prob": "",
            "replay_pct": "",
            "n_samples": "",
            "elapsed_s": f"{elapsed:.1f}",
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_continual(base_checkpoint: str, run: dict, n_samples: int, suffix: str = "", skip_generation: bool = False) -> None:
    model_name = run["model-name"] + suffix
    checkpoint_out = f"models/{TODAY}_{model_name}.pth"

    cmd = [
        sys.executable, "scripts/continual_train.py",
        "--checkpoint", base_checkpoint,
        "--strategy", str(run["strategy"]),
        "--grad-update-prob", str(run["grad-update-prob"]),
        "--replay-pct", str(run["replay-pct"]),
        "--new-label", NEW_LABEL,
        "--new-display-name", NEW_LABEL.replace("_", " "),
        "--new-action", NEW_ACTION,
        "--new-url", NEW_URL,
        "--new-browser", NEW_BROWSER,
        "--model-name", model_name,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--n-samples", str(n_samples),
    ]
    if skip_generation:
        cmd.append("--skip-generation")

    print(f"\n{'='*60}")
    print(f"[RUN] {model_name}  (n_samples={n_samples})")
    print(f"{'='*60}")
    t_start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t_start
        print(f"[TIMING] {model_name}: {elapsed:.1f}s")

        run_evaluate(checkpoint_out, TEST_DIR_CONTINUAL)
        record_result(
            {
                "model_name": model_name,
                "type": "continual",
                "base_model": Path(base_checkpoint).stem,
                "strategy": run["strategy"],
                "grad_update_prob": run["grad-update-prob"],
                "replay_pct": run["replay-pct"],
                "n_samples": n_samples,
                "elapsed_s": f"{elapsed:.1f}",
                "error": "",
            },
            checkpoint_out,
        )
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"[ERROR] {model_name} failed after {elapsed:.1f}s: {e}")
        results_rows.append({
            "model_name": model_name,
            "type": "continual",
            "base_model": Path(base_checkpoint).stem,
            "strategy": run["strategy"],
            "grad_update_prob": run["grad-update-prob"],
            "replay_pct": run["replay-pct"],
            "n_samples": n_samples,
            "elapsed_s": f"{elapsed:.1f}",
            "error": str(e),
        })
        _flush_results_csv()
        print(f"[RESULTS] Failure recorded, continuing to next run.")


# ── Train base models ─────────────────────────────────────────────────────────

try:
    BASE_MODEL_A_PATH = run_base_model("baseModelA", TRAINING_DATASET_150)
except Exception as e:
    print(f"[ERROR] baseModelA failed: {e}")
    raise SystemExit(1)

try:
    BASE_MODEL_B_PATH = run_base_model("baseModelB", TRAINING_DATASET_300)
except Exception as e:
    print(f"[ERROR] baseModelB failed: {e}")
    raise SystemExit(1)

# ── Continual training: Base Model A + 150 new files ─────────────────────────

print("\n" + "="*60)
print("Continual training on Base Model A — 150 new files")
print("="*60)
for i, run in enumerate(RUNS):
    run_continual(BASE_MODEL_A_PATH, run, n_samples=NEW_DATA_150, suffix="_A150", skip_generation=(i > 0))

# ── Continual training: Base Model A + 300 new files ─────────────────────────

print("\n" + "="*60)
print("Continual training on Base Model A — 300 new files")
print("="*60)
for i, run in enumerate(RUNS):
    run_continual(BASE_MODEL_A_PATH, run, n_samples=NEW_DATA_300, suffix="_A300", skip_generation=(i > 0))
