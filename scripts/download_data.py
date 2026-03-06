"""Download the LIAR dataset zip archive and extract raw TSV splits."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/dataset.yaml")
LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
ARCHIVE_SPLIT_NAMES = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "test": "test.tsv",
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
    """Build local output paths for all LIAR splits.

    Args:
        config: Parsed dataset configuration dictionary.

    Returns:
        Mapping from split names to local TSV output paths.
    """

    liar_cfg = config["liar"]
    return {
        "train": Path(liar_cfg["train_path"]),
        "valid": Path(liar_cfg["valid_path"]),
        "test": Path(liar_cfg["test_path"]),
    }


def download_archive(url: str = LIAR_ZIP_URL) -> zipfile.ZipFile:
    """Download the LIAR zip archive and return it as an in-memory ZipFile.

    Args:
        url: Remote URL for the LIAR dataset zip archive.

    Returns:
        In-memory zip archive containing the LIAR split files.
    """

    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def find_archive_member(archive: zipfile.ZipFile, filename: str) -> str:
    """Find a zip member by its basename.

    Args:
        archive: Open LIAR dataset zip archive.
        filename: Expected TSV filename inside the archive.

    Returns:
        The matching archive member name.

    Raises:
        FileNotFoundError: If the expected file is not present in the archive.
    """

    for member in archive.namelist():
        if Path(member).name == filename:
            return member
    raise FileNotFoundError(f"{filename} was not found in the LIAR zip archive.")


def extract_split(archive: zipfile.ZipFile, archive_name: str, output_path: Path) -> None:
    """Extract one TSV split from the zip archive to the configured output path.

    Args:
        archive: Open LIAR dataset zip archive.
        archive_name: Expected TSV filename inside the archive.
        output_path: Destination path for the extracted TSV file.
    """

    member = find_archive_member(archive, archive_name)
    output_path.write_bytes(archive.read(member))
    logger.info("Saved %s", output_path)


def main() -> None:
    """Download the LIAR zip archive and extract any missing split files."""

    config = load_config()
    split_paths = get_split_paths(config)
    output_dir = split_paths["train"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_splits: dict[str, Path] = {}
    for split, output_path in split_paths.items():
        if output_path.exists():
            logger.info("%s already exists, skipping.", output_path)
            continue
        missing_splits[split] = output_path

    if not missing_splits:
        logger.info("All LIAR split files already exist.")
        return

    logger.info("Downloading LIAR zip from UCSB.")
    archive = download_archive()

    with archive:
        for split, output_path in missing_splits.items():
            extract_split(archive, ARCHIVE_SPLIT_NAMES[split], output_path)


if __name__ == "__main__":
    main()
