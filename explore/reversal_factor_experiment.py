"""Isolated reversal-factor experiment; never changes the frozen GMI-ADG product."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from product_evaluation import load_product_context


END = pd.Timestamp("2026-07-31")
CAPITAL = 1_000_000.0
OUTPUT = ROOT / "explore" / "report_work" / "reversal_factor_experiment"
VARIANTS = {
    "GMI-ADG": 0.0,
    "REV-0.5": 0.5,
    "REV-1.0": 1.0,
    "REV-2.0": 2.0,
}
PERIODS = {
    "Full History": (pd.Timestamp("2015-09-07"), END),
    "In Sample": (pd.Timestamp("2015-09-07"), pd.Timestamp("2024-12-31")),
    "Historical Validation": (pd.Timestamp("2025-01-01"), END),
}


def one_month_reversal(history: pd.DataFrame) -> pd.Series:
    """High score means weak return over the latest 21 valuation days."""
    return -((1.0 + history.iloc[-21:]).prod() - 1.0)


def run_experiment():
    context, sora = load_product_context(END, CAPITAL)
    original_momentum = context["momentum_12_1"]
    results = {}

    for label, gamma in VARIANTS.items():
        def blended_score(history, coefficient=gamma):
            return original_momentum(history) + coefficient * one_month_reversal(history)

        context["momentum_12_1"] = original_momentum if gamma == 0 else blended_score
        result = context["run_dynamic_attack_backtest"](
            name=label, end_date=END, sora_frame=sora, allow_holdout=False
        )
        results[label] = result

    context["momentum_12_1"] = original_momentum
    rows = []
    for period, (start, end) in PERIODS.items():
        for label, result in results.items():
            metric = context["compute_metrics"](result, start, end)
            rows.append({
                "Period": period,
                "Variant": label,
                "Reversal Coefficient": VARIANTS[label],
                "Cumulative Return": metric["Cumulative Return"],
                "Annualized Return": metric["Annualized Return"],
                "Annualized Volatility": metric["Annualized Volatility"],
                "Sharpe Ratio": metric["Sharpe Ratio"],
                "Sortino Ratio": metric["Sortino Ratio"],
                "Maximum Drawdown": metric["Maximum Drawdown"],
                "Annual Two-Way Traded Notional": metric["Annual Two-Way Traded Notional"],
                "Number of Rebalances": metric["Number of Rebalances"],
            })
    summary = pd.DataFrame(rows)

    full_returns = pd.concat(
        {label: result["returns"].loc[PERIODS["Full History"][0]:END]
         for label, result in results.items()}, axis=1
    ).dropna()
    wealth = (1.0 + full_returns).cumprod()
    correlations = full_returns.corr()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "reversal_factor_comparison.csv", index=False)
    full_returns.to_csv(OUTPUT / "daily_returns.csv", index_label="Date")
    correlations.to_csv(OUTPUT / "return_correlations.csv")

    colors = {"GMI-ADG": "#b82626", "REV-0.5": "#2b549c",
              "REV-1.0": "#e58b2a", "REV-2.0": "#55b7a7"}
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for label in wealth:
        ax.plot(wealth.index, wealth[label] - 1.0, label=label,
                color=colors[label], linewidth=2.4 if label == "GMI-ADG" else 1.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(False)
    ax.set_title("Reversal-Factor Experiment: Cumulative Net Return\n2015-09-07 to 2026-07-31")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "reversal_factor_cumulative_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary, correlations


if __name__ == "__main__":
    comparison, return_correlations = run_experiment()
    columns = ["Period", "Variant", "Annualized Return", "Sharpe Ratio",
               "Maximum Drawdown", "Annual Two-Way Traded Notional"]
    print(comparison[columns].to_string(index=False))
    print("\nOutput:", OUTPUT)
