"""
CLI entry point for BarkAI training data generation.

Reads command definitions from config/commands.yaml and generates synthetic
audio for each command using Bark AI.

Usage:
    # Generate all commands defined in commands.yaml
    python scripts/generate_data.py

    # Generate only specific commands (label keys from commands.yaml)
    python scripts/generate_data.py --commands Get_Me_Gmail --commands New_Word_Document

    # Override defaults
    python scripts/generate_data.py --n-samples 500 --data-dir data --config config/commands.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Resolve src/ package from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_gen.data_generator import generate_audio_for_command, load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_commands(config_path: Path) -> dict:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("commands", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BarkAI training audio for voice commands.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/commands.yaml"),
        help="Path to commands.yaml (default: config/commands.yaml)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root output directory; one subfolder per command label (default: data/)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Target number of .wav files per command (default: 2000)",
    )
    parser.add_argument(
        "--commands", action="append", metavar="LABEL", default=None,
        help="Generate only this label (repeat to specify multiple). Omit to generate all.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    all_commands = _load_commands(args.config)
    if not all_commands:
        logger.error("No commands found in %s", args.config)
        sys.exit(1)

    # Filter to requested labels if --commands was provided
    if args.commands:
        unknown = set(args.commands) - set(all_commands)
        if unknown:
            logger.error("Unknown command labels: %s. Available: %s", unknown, list(all_commands))
            sys.exit(1)
        to_generate = {k: all_commands[k] for k in args.commands}
    else:
        to_generate = all_commands

    logger.info("Commands to generate: %s", list(to_generate))
    logger.info("Target samples per command: %d", args.n_samples)
    logger.info("Output directory: %s", args.data_dir)

    load_models()

    results = {}
    for label, meta in to_generate.items():
        display_name = meta.get("display_name", label.replace("_", " "))
        count = generate_audio_for_command(
            label=label,
            display_name=display_name,
            n_samples=args.n_samples,
            out_dir=args.data_dir,
            seed=args.seed,
        )
        results[label] = count

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    for label, count in results.items():
        print(f"  {label}: {count} files")
    print(f"Output: {args.data_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
