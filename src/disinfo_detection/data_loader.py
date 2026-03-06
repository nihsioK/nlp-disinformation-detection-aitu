"""Utilities for loading LIAR dataset splits from local TSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml


VALID_SPLITS = {"train", "valid", "test"}


def load_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load and return the dataset config as a dict.

    Args:
        config_path: Path to the dataset YAML configuration file.

    Returns:
        Parsed YAML configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found at {path}.")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_liar(split: str, config_path: str = "config/dataset.yaml") -> pd.DataFrame:
    """Load a LIAR split into a DataFrame.

    Args:
        split: One of 'train', 'valid', 'test'.
        config_path: Path to dataset.yaml.

    Returns:
        DataFrame with all 14 LIAR columns plus 'label_id' (int, 0-5).

    Raises:
        FileNotFoundError: If split file does not exist.
        ValueError: If split is not 'train', 'valid', or 'test'.
    """

    if split not in VALID_SPLITS:
        raise ValueError("split must be one of 'train', 'valid', or 'test'.")

    config = load_config(config_path)
    liar_cfg = config["liar"]
    path = Path(liar_cfg[f"{split}_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"LIAR {split} file not found at {path}. Run scripts/download_data.py first."
        )

    frame = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    frame.columns = liar_cfg["columns"]
    label_map = liar_cfg["label_map"]
    frame["label_id"] = frame["label"].map(label_map)

    if frame["label_id"].isna().any():
        unknown_labels = sorted(frame.loc[frame["label_id"].isna(), "label"].dropna().unique())
        raise ValueError(f"Found labels not present in config label_map: {unknown_labels}")

    frame["label_id"] = frame["label_id"].astype(int)
    return frame


def get_label_map(config_path: str = "config/dataset.yaml") -> dict:
    """Return string-to-integer label mapping from config.

    Args:
        config_path: Path to dataset.yaml.

    Returns:
        Dictionary mapping LIAR label strings to integer ids.
    """

    return load_config(config_path)["liar"]["label_map"]


def get_splits(config_path: str = "config/dataset.yaml") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the train, validation, and test LIAR DataFrames.

    Args:
        config_path: Path to dataset.yaml.

    Returns:
        Tuple of `(train_df, valid_df, test_df)`.
    """

    return (
        load_liar("train", config_path=config_path),
        load_liar("valid", config_path=config_path),
        load_liar("test", config_path=config_path),
    )
