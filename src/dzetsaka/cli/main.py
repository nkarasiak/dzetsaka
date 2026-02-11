"""Command-line interface for dzetsaka."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional

from dzetsaka.application.use_cases.classify_raster import run_classification
from dzetsaka.application.use_cases.train_model import run_training
from dzetsaka.logging import show_error_dialog


class _CLIProgress:
    """Simple stdout progress helper for CLI users."""

    def __init__(self) -> None:
        self._last_text: str = ""
        self._last_percent: int = -1

    def setProgress(self, value: float | int) -> None:
        percent = max(0, min(100, int(float(value))))
        if percent != self._last_percent:
            self._last_percent = percent
            print(f"[dzetsaka] progress {percent}%", flush=True)

    def setProgressText(self, text: str) -> None:
        message = text.strip()
        if message and message != self._last_text:
            self._last_text = message
            print(f"[dzetsaka] {message}", flush=True)


def _parse_split_config(value: str) -> str | int | float:
    normalized = value.strip()
    if not normalized:
        return 100
    if normalized.upper() in {"SLOO", "STAND"}:
        return normalized.upper()
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return normalized


def _parse_extra_params(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return {}
    trimmed = value.strip()
    if not trimmed:
        return {}
    if trimmed.startswith("@"):
        file_path = Path(trimmed[1:])
        if not file_path.exists():
            raise ValueError(f"Extra params file not found: {file_path}")
        return json.loads(file_path.read_text(encoding="utf-8"))
    return json.loads(trimmed)


def _run_classify(args: argparse.Namespace) -> int:
    progress = _CLIProgress()
    try:
        run_classification(
            raster_path=args.raster,
            model_path=args.model,
            output_path=args.output,
            mask_path=args.mask,
            confidence_map=args.confidence,
            nodata=args.nodata,
            feedback=progress,
        )
        print(f"Classification output written to {args.output}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error
        show_error_dialog("dzetsaka CLI Error", exc)
        return 1


def _run_train(args: argparse.Namespace) -> int:
    progress = _CLIProgress()
    try:
        extra = _parse_extra_params(args.extra)
        split_config = _parse_split_config(args.split_config)
        run_training(
            raster_path=args.raster,
            vector_path=args.vector,
            class_field=args.class_field,
            model_path=args.model,
            split_config=split_config,
            random_seed=args.random_seed,
            matrix_path=args.matrix_path,
            classifier=args.classifier,
            extra_params=extra,
            feedback=progress,
        )
        print(f"Model trained and saved to {args.model}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error
        show_error_dialog("dzetsaka CLI Error", exc)
        return 1


def _configure_cli() -> ArgumentParser:
    parser = argparse.ArgumentParser(prog="dzetsaka", description="dzetsaka core CLI")
    subparsers = parser.add_subparsers(dest="command")

    classify_parser = subparsers.add_parser("classify", help="Run classification from CLI")
    classify_parser.add_argument("--raster", required=True, help="Path to input raster")
    classify_parser.add_argument("--model", required=True, help="Path to trained model")
    classify_parser.add_argument("--output", required=True, help="Path to output raster")
    classify_parser.add_argument("--mask", help="Optional training mask raster")
    classify_parser.add_argument("--confidence", help="Optional confidence raster output")
    classify_parser.add_argument("--nodata", type=int, default=-9999, help="NODATA value to use")
    classify_parser.set_defaults(func=_run_classify)

    train_parser = subparsers.add_parser("train", help="Train model from raster/shapefile")
    train_parser.add_argument("--raster", required=True, help="Path to training raster")
    train_parser.add_argument("--vector", required=True, help="Path to training shapefile")
    train_parser.add_argument("--class-field", default="Class", help="Class field name")
    train_parser.add_argument("--model", required=True, help="Path to save trained model")
    train_parser.add_argument("--split-config", default="100", help="Split percent or SLOO/STAND")
    train_parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    train_parser.add_argument("--matrix-path", help="Optional path for confusion matrix")
    train_parser.add_argument("--classifier", default="GMM", help="Classifier code (RF, XGB, etc.)")
    train_parser.add_argument(
        "--extra",
        help="JSON string or @path to JSON file with extra parameters",
    )
    train_parser.set_defaults(func=_run_train)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _configure_cli()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
