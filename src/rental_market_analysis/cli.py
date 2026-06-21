from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import DEFAULT_OUTPUT, DEFAULT_SOURCES, save_combined_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the cleaned Zurich vs Milan rental comparison dataset.",
    )
    parser.add_argument(
        "--zurich",
        type=Path,
        default=DEFAULT_SOURCES["Zurich"],
        help="Path to the raw Zurich CSV extract.",
    )
    parser.add_argument(
        "--milan",
        type=Path,
        default=DEFAULT_SOURCES["Milan"],
        help="Path to the raw Milan CSV extract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the cleaned CSV dataset.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_path = save_combined_dataset(
        output_path=args.output,
        source_files={"Zurich": args.zurich, "Milan": args.milan},
    )
    print(f"Clean dataset saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
