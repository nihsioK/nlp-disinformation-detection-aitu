"""Package the paper source and final figures for Overleaf."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "overleaf_submission.zip"
FIGURE_PATHS = (
    "figures/per_class_f1_grouped.pdf",
    "figures/confusion_matrices.pdf",
    "figures/training_curves.pdf",
)
README_NAME = "README_OVERLEAF.md"
README_CONTENT = """# Overleaf Build Notes

Set `main.tex` as the main file in Overleaf.

Recommended compiler: pdfLaTeX. Run LaTeX -> BibTeX -> LaTeX -> LaTeX if
Overleaf does not do this automatically.

This archive intentionally contains only the manuscript source, bibliography,
and final paper figures. Training data, model checkpoints, prediction artifacts,
and local build outputs are not needed for Overleaf compilation.
"""


def validate_overleaf_inputs(
    project_root: Path,
    tex_path: Path,
    bib_path: Path,
    figure_paths: Sequence[str] = FIGURE_PATHS,
) -> list[Path]:
    """Validate files required for the Overleaf package."""

    required = [tex_path, bib_path, *(project_root / path for path in figure_paths)]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Overleaf package inputs: " + ", ".join(map(str, missing))
        )
    return required


def create_overleaf_package(
    project_root: Path = PROJECT_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    tex_filename: str = "main.tex",
    bib_filename: str = "references.bib",
    figure_paths: Sequence[str] = FIGURE_PATHS,
) -> Path:
    """Create an Overleaf-ready zip containing manuscript source and figures."""

    tex_path = project_root / tex_filename
    bib_path = project_root / bib_filename
    validate_overleaf_inputs(project_root, tex_path, bib_path, figure_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(tex_path, "main.tex")
        archive.write(bib_path, "references.bib")
        for figure_path in figure_paths:
            archive.write(project_root / figure_path, figure_path)
        archive.writestr(README_NAME, README_CONTENT)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing main.tex, references.bib, and figures/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output zip path for the Overleaf package.",
    )
    parser.add_argument(
        "--tex",
        default="main.tex",
        help="Manuscript TeX filename to package as main.tex.",
    )
    parser.add_argument(
        "--bib",
        default="references.bib",
        help="Bibliography filename to package as references.bib.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output = create_overleaf_package(
        project_root=args.project_root.resolve(),
        output_path=args.output.resolve(),
        tex_filename=args.tex,
        bib_filename=args.bib,
    )
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
