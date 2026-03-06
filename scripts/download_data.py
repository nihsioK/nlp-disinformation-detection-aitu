"""Download the LIAR dataset and persist raw splits as TSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/dataset.yaml")
LIAR_URLS = {
    "train": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/train.tsv",
    "valid": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/valid.tsv",
    "test": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/test.tsv",
}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load dataset configuration from YAML.

    Args:
        config_path: Path to the dataset configuration file.

    Returns:
        Parsed dataset configuration dictionary.
    """

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_split_paths(config: dict) -> dict[str, Path]:
    """Build output paths for all LIAR splits.

    Args:
        config: Parsed dataset configuration dictionary.

    Returns:
        Mapping from project split names to local TSV paths.
    """

    liar_cfg = config["liar"]
    return {
        "train": Path(liar_cfg["train_path"]),
        "valid": Path(liar_cfg["valid_path"]),
        "test": Path(liar_cfg["test_path"]),
    }


def download_split(url: str, output_path: Path) -> None:
    """Download one LIAR TSV split and save it to disk.

    Args:
        url: Remote TSV download URL.
        output_path: Destination TSV path.
    """

    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    logger.info("Saved split to %s", output_path)


def main() -> None:
    """Download LIAR and save the train, validation, and test TSV files."""

    config = load_config()
    split_paths = get_split_paths(config)
    output_dir = split_paths["train"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading LIAR TSV files from Hugging Face.")
    for split, output_path in split_paths.items():
        if output_path.exists():
            logger.info("%s already exists, skipping.", output_path)
            continue
        download_split(LIAR_URLS[split], output_path)


if __name__ == "__main__":
    main()
