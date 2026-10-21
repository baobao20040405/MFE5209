"""Synchronize modular English source into the detailed English notebook."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "MFE5209_Global_Multi_Asset_Momentum_and_Risk_Allocation.ipynb"
MODULES = [
    "config.py",
    "data_pipeline.py",
    "models.py",
    "strategies.py",
    "reporting.py",
    "stress_tests.py",
    "benchmarks.py",
    "robustness.py",
    "attribution.py",
]
MARKER = re.compile(
    r"^# --- Original Notebook code (?:unit|module) (\d+)\s*-+\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def normalize_markdown_headings(source: str) -> str:
    """Repair malformed translated headings while preserving prose and equations."""
    normalized_lines = []
    for line in source.splitlines():
        # Examples repaired here: ``##3.1``, ``## # 5.1`` and ``# # # 8.1``.
        heading = re.match(r"^\s*(?:#\s*){1,6}(.*)$", line)
        if heading:
            title = heading.group(1).strip()
            numbered = re.match(r"^(\d+)(?:\.(\d+))?\b", title)
            if numbered:
                level = 2 if numbered.group(2) is None else 3
                line = f"{'#' * level} {title}"
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def module_code_segments() -> list[str]:
    segments: list[str] = []
    for filename in MODULES:
        text = (ROOT / filename).read_text(encoding="utf-8")
        matches = list(MARKER.finditer(text))
        if not matches:
            raise RuntimeError(f"No notebook cell markers found in {filename}.")
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            source = text[match.end():end].lstrip("\r\n")
            # Module files retain the next notebook section as Python comments
            # immediately before the next cell marker.  Exclude those comments
            # from the preceding code cell so headings appear only in Markdown.
            following_heading = re.search(
                r"\n#\s+(?:(?:#{1,4}\s+)?\d+\.\d+(?:\.\d+)?(?:\s+|$)|Part\s+[IVX]+\b)",
                source,
                flags=re.IGNORECASE,
            )
            if following_heading:
                source = source[:following_heading.start()]
            segments.append(source.rstrip() + "\n")
    return segments


def build_notebook() -> None:
    if not OUTPUT.exists():
        raise FileNotFoundError(
            "The detailed English notebook template is missing. Restore it before synchronization."
        )
    notebook = nbformat.read(OUTPUT, as_version=4)
    segments = module_code_segments()
    ordinary_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and not cell.get("metadata", {}).get("visual_diagnostics_appendix")
        and not cell.get("metadata", {}).get("product_evaluation_appendix")
    ]
    if len(segments) != len(ordinary_cells):
        raise RuntimeError(
            f"Module segments ({len(segments)}) do not match notebook code cells "
            f"({len(ordinary_cells)})."
        )

    segment_index = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = normalize_markdown_headings(cell.source)
            continue
        if cell.cell_type != "code" or cell.get("metadata", {}).get(
            "visual_diagnostics_appendix"
        ) or cell.get("metadata", {}).get("product_evaluation_appendix"):
            continue
        source = segments[segment_index]
        segment_index += 1
        if cell.source.rstrip() != source.rstrip():
            cell.source = source
            cell.outputs = []
            cell.execution_count = None

    if not any(c.get("metadata", {}).get("product_evaluation_appendix") for c in notebook.cells):
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            "## Part X. Final Product Evaluation and Course Data Export\n\n"
            "### 10.1 Historical validation and the October test\n\n"
            "The final product is GMI-ATC (GMI-ADG plus the 20-day trend confirmation of Section 3.3.1). The planned formal evaluation is "
            "2026-10-01 through 2026-10-31. Before the period finishes, the default "
            "evaluation returns PENDING. Historical validation can be reported "
            "separately, with its model-selection limitations disclosed. Update "
            "SORA_daily.csv through the evaluation endpoint before running.\n\n"
            "### 10.2 One-call evaluation and closing-price export\n\n"
            "The function below replays existing holdings and execution state, "
            "then evaluates only the requested interval. It exports metrics, "
            "FIFO trades, daily NAV/P&L, weights and a wealth chart. Price exports "
            "include all 18 assets, unadjusted local closes, adjusted local "
            "closes and adjusted SGD closes.\n\n"
            "After October ends, call `evaluate_final_product()`. For a partial "
            "observation use `evaluate_final_product(allow_partial=True)`.\n\n"
            "To export prices already loaded by this notebook, call "
            "`export_asset_closing_prices(globals(), 'submission_data', end=RESEARCH_END)`."
        ))
        notebook.cells.append(nbformat.v4.new_code_cell(
            "from product_evaluation import evaluate_final_product, export_asset_closing_prices\n"
            "# Run after the formal test period ends and SORA has been updated:\n"
            "# product = evaluate_final_product(start='2026-10-01', end='2026-10-31')\n"
            "# Historical validation (report separately from the October test):\n"
            "# historical = evaluate_final_product(start='2025-01-01', end='2026-07-31',\n"
            "#     output_dir='historical_product_outputs')\n"
            "# Export data from the current notebook session:\n"
            "# price_files = export_asset_closing_prices(globals(), 'submission_data', end=RESEARCH_END)\n",
            metadata={"product_evaluation_appendix": True},
        ))
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## Part X. Final Product Evaluation'):
            cell.source = cell.source.replace(
                'Update SORA_daily.csv through the evaluation endpoint before running.',
                'Update `data/raw/raw sora.csv` in the parent project folder with the unedited MAS export before running the final product.')
            cell.source = cell.source.replace(
                'It exports metrics, FIFO trades, daily NAV/P&L, weights and a wealth chart.',
                'It delegates to the same standalone final-product evaluator as `code/main.py`, exporting four-series metrics, FIFO trades, daily NAV/P&L, weights and comparison figures. Observed exchange-session quotes are checked before a full test can be called complete.')
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(
        f"Synchronized {len(segments)} code cells in {OUTPUT.name}; "
        f"preserved {sum(c.cell_type == 'markdown' for c in notebook.cells)} Markdown cells."
    )


if __name__ == "__main__":
    build_notebook()
