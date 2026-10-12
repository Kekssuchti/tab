"""Save descriptive LaTeX tables for both filtered clinical datasets.

Run via src/plotting/recreate_all_tables.sh, or:
    uv run python -m src.plotting.descriptive_tables --kind normal

Default: main and readmission cohort tables under tables/descriptive/.
Each fragment contains a complete table environment and requires booktabs and
graphicx. Edit feature labels/order in src.exploration.descriptive_table.
Generated aggregates remain local and gitignored. No patient rows are exported.
"""

import argparse
from pathlib import Path

from src.config import config
from src.plotting.utils.descriptive_table import descriptive_statistics_table_to_latex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("normal", "readmission", "all"), default="all")
    parser.add_argument("--filtered-dir", type=Path, default=config.dir_data / "filtered")
    parser.add_argument("--output-dir", type=Path, default=config.dir_root / "tables" / "descriptive")
    parser.add_argument("--stdout", action="store_true", help="Also print the generated LaTeX.")
    args = parser.parse_args()
    kinds = ("normal", "readmission") if args.kind == "all" else (args.kind,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        latex = descriptive_statistics_table_to_latex(args.filtered_dir, type=kind)
        path = args.output_dir / f"features_{kind}.tex"
        path.write_text(latex + "\n", encoding="utf-8")
        print(f"table: {path}")
        if args.stdout:
            print(latex)


if __name__ == "__main__":
    main()
