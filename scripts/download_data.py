"""Download the LIAR dataset and persist raw splits as TSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/dataset.yaml")

HF_SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    liar_cfg = config["liar"]
    columns = liar_cfg["columns"]

    split_paths = {
        "train": Path(liar_cfg["train_path"]),
        "valid": Path(liar_cfg["valid_path"]),
        "test":  Path(liar_cfg["test_path"]),
    }

    output_dir = split_paths["train"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading LIAR dataset from HuggingFace (ucsbnlp/liar)...")
    ds = load_dataset("ucsbnlp/liar")

    for split, out_path in split_paths.items():
        if out_path.exists():
            logger.info("%s already exists, skipping.", out_path)
            continue

        hf_split = HF_SPLIT_MAP[split]
        df = ds[hf_split].to_pandas()

        # Keep only the columns we need, in the right order
        df = df[[c for c in columns if c in df.columns]]

        df.to_csv(out_path, sep="\t", index=False, header=False)
        logger.info("Saved %s — %d rows, %d cols", out_path, len(df), len(df.columns))

    logger.info("Done. All splits saved to %s/", output_dir)


if __name__ == "__main__":
    main()