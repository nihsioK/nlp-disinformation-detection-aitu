"""Download the LIAR dataset and persist raw splits as TSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/dataset.yaml")


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
        Mapping from Hugging Face split names to local TSV paths.
    """

    liar_cfg = config["liar"]
    return {
        "train": Path(liar_cfg["train_path"]),
        "validation": Path(liar_cfg["valid_path"]),
        "test": Path(liar_cfg["test_path"]),
    }


def save_split(dataset: Any, split: str, output_path: Path, columns: list[str]) -> None:
    """Save one dataset split to a tab-separated file without headers.

    Args:
        dataset: Loaded LIAR dataset collection.
        split: Split name to export.
        output_path: Destination TSV path.
        columns: Column order required by the project.
    """

    frame = pd.DataFrame(dataset[split])[columns]
    frame.to_csv(output_path, sep="\t", header=False, index=False)
    logger.info("Saved %s split to %s", split, output_path)


def main() -> None:
    """Download LIAR and save the train, validation, and test TSV files."""

    config = load_config()
    split_paths = get_split_paths(config)
    columns = config["liar"]["columns"]
    output_dir = split_paths["train"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if all(path.exists() for path in split_paths.values()):
        logger.info("Data already exists, skipping.")
        return

    from datasets import load_dataset

    logger.info("Downloading LIAR dataset from Hugging Face.")
    dataset = load_dataset("liar")

    for split, output_path in split_paths.items():
        save_split(dataset, split, output_path, columns)


if __name__ == "__main__":
    main()
