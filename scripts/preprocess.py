"""Run the full LIAR preprocessing pipeline and persist processed splits."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.disinfo_detection.data_loader import get_splits, load_config
from src.disinfo_detection.preprocessing import preprocess_dataframe


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def process_and_save_split(split_name: str, frame: pd.DataFrame, output_dir: Path) -> None:
    """Preprocess a DataFrame split and save it as a pickle file.

    Args:
        split_name: Dataset split name.
        frame: Raw LIAR DataFrame for the split.
        output_dir: Directory where processed pickles will be written.
    """

    start_time = time.perf_counter()
    processed = preprocess_dataframe(frame)
    output_path = output_dir / f"{split_name}.pkl"
    processed.to_pickle(output_path)
    elapsed = time.perf_counter() - start_time
    logger.info(
        "Processed %s split with %d rows in %.2f seconds and saved to %s",
        split_name,
        len(processed),
        elapsed,
        output_path,
    )


def main() -> None:
    """Load all raw splits, preprocess them, and save pickle artifacts."""

    config = load_config()
    output_dir = Path(config["liar"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = get_splits()
    process_and_save_split("train", train_df, output_dir)
    process_and_save_split("valid", valid_df, output_dir)
    process_and_save_split("test", test_df, output_dir)


if __name__ == "__main__":
    main()
