"""Research compatibility entry: always delegate to the submitted product.

The research notebook may import this name from explore/. Keep no independent
evaluation, cost, SORA, output-publication or price-export implementation here.
"""
from pathlib import Path
import importlib.util

_entry = Path(__file__).resolve().parents[1] / 'product_evaluation.py'
_spec = importlib.util.spec_from_file_location('_gmi_research_product_bridge', _entry)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

evaluate_final_product = _module.evaluate_final_product
export_asset_closing_prices = _module.export_asset_closing_prices
cli = _module.cli

if __name__ == '__main__':
    cli()
