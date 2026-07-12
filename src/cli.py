"""
cli.py
======
Entry-point for the ``supportai`` command-line interface.

Sub-commands will be added in later phases (data prep, training, inference).
Running ``supportai`` with no arguments prints help.

Usage::

    supportai --help
    supportai env-check          # Phase 1 - environment diagnostics
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supportai",
        description=(
            "SupportAI - lightweight customer-support ticket routing system.\n"
            "Run 'supportai <command> --help' for sub-command details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ------------------------------------------------------------------
    # env-check sub-command (Phase 1)
    # ------------------------------------------------------------------
    env_parser = sub.add_parser(
        "env-check",
        help="Check Python version, installed packages, and hardware.",
    )
    env_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed version info for every package.",
    )

    return parser


def _cmd_env_check(args: argparse.Namespace) -> int:
    """Run environment diagnostics and return exit code."""
    from src.utils.logging_utils import get_logger

    logger = get_logger(__name__)

    import platform
    import sys

    import numpy as np
    import sklearn
    import torch

    logger.info("=== SupportAI Environment Check ===")
    logger.info("Python       : %s", sys.version)
    logger.info("Platform     : %s", platform.platform())
    logger.info("PyTorch      : %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("NumPy        : %s", np.__version__)
    logger.info("Scikit-learn : %s", sklearn.__version__)

    if args.verbose:
        import pandas as pd
        import transformers

        logger.info("Pandas       : %s", pd.__version__)
        logger.info("Transformers : %s", transformers.__version__)

    from src.utils.constants import BASE_DIR, CONFIG_DIR, OUTPUT_DIR

    logger.info("BASE_DIR     : %s", BASE_DIR)
    logger.info("OUTPUT_DIR   : %s", OUTPUT_DIR)
    logger.info("CONFIG_DIR   : %s", CONFIG_DIR)
    logger.info("=== Check complete - all imports succeeded ===")
    return 0


def main() -> None:
    """CLI entry-point registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "env-check": _cmd_env_check,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = dispatch.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command!r}")

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
