"""Download the default Qwen Instruct checkpoint for Kev. Does not run inference."""

from __future__ import annotations

import argparse
import logging
import sys

from kev.backends import MLX_DEFAULT_REPO

logger = logging.getLogger("kev.download")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a Qwen Instruct checkpoint for Kev.")
    parser.add_argument(
        "--repo",
        default=MLX_DEFAULT_REPO,
        help="Hugging Face repo id (default: 1.5B 4-bit MLX)",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            from transformers.utils.hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required to download weights. "
                "Install huggingface_hub or use --model mock."
            ) from exc
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("downloading repo=%s", args.repo)
    path = snapshot_download(repo_id=args.repo)
    logger.info("cached path=%s", path)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
