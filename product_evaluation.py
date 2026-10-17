"""Compatibility entry point. The final implementation lives in code/."""
from pathlib import Path
import importlib.util
import sys

code_path = Path(__file__).resolve().parent / 'code'
sys.path.insert(0, str(code_path))
spec = importlib.util.spec_from_file_location('_gmi_final_product', code_path / 'product_evaluation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluate_final_product = module.evaluate_final_product
export_asset_closing_prices = module.export_asset_closing_prices
cli = module.cli

if __name__ == '__main__':
    module.cli()
