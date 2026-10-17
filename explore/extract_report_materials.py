from __future__ import annotations

import base64
import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "MFE5209_Global_Multi_Asset_Momentum_and_Risk_Allocation.ipynb"
OUTPUT = ROOT / "report_work"
ASSETS = OUTPUT / "assets"
TABLES = OUTPUT / "tables"


def safe_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", value)
    return value[:80] or "untitled"


def cell_heading(source: str) -> str:
    for line in source.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return safe_name(line)
    return "cell"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    index_rows: list[dict[str, object]] = []

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        heading = cell_heading(source)
        image_no = 0
        table_no = 0
        text_no = 0
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            png = data.get("image/png")
            if png:
                image_no += 1
                raw = "".join(png) if isinstance(png, list) else png
                path = ASSETS / f"cell_{cell_index:03d}_img_{image_no:02d}_{heading}.png"
                path.write_bytes(base64.b64decode(raw))
                index_rows.append({
                    "cell": cell_index,
                    "output": output_index,
                    "kind": "image",
                    "path": str(path),
                    "source_preview": source[:250].replace("\n", " "),
                })

            html = data.get("text/html")
            if html:
                html_text = "".join(html) if isinstance(html, list) else html
                try:
                    frames = pd.read_html(StringIO(html_text))
                except (ValueError, ImportError):
                    frames = []
                for frame in frames:
                    table_no += 1
                    path = TABLES / f"cell_{cell_index:03d}_table_{table_no:02d}_{heading}.csv"
                    frame.to_csv(path, index=False, encoding="utf-8-sig")
                    index_rows.append({
                        "cell": cell_index,
                        "output": output_index,
                        "kind": "table",
                        "path": str(path),
                        "source_preview": source[:250].replace("\n", " "),
                    })

            text = data.get("text/plain") or output.get("text")
            if text:
                text_no += 1
                text_value = "".join(text) if isinstance(text, list) else text
                path = TABLES / f"cell_{cell_index:03d}_text_{text_no:02d}_{heading}.txt"
                path.write_text(str(text_value), encoding="utf-8")
                index_rows.append({
                    "cell": cell_index,
                    "output": output_index,
                    "kind": "text",
                    "path": str(path),
                    "source_preview": source[:250].replace("\n", " "),
                })

    pd.DataFrame(index_rows).to_csv(OUTPUT / "material_index.csv", index=False, encoding="utf-8-sig")
    print(f"Extracted {sum(row['kind'] == 'image' for row in index_rows)} images")
    print(f"Extracted {sum(row['kind'] == 'table' for row in index_rows)} tables")
    print(f"Index: {OUTPUT / 'material_index.csv'}")


if __name__ == "__main__":
    main()
