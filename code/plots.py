"""Consistent final-product comparisons using the original colors and fonts."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


def _dates(ax, dates):
    ticks = pd.date_range(dates[0], dates[-1], periods=min(5, len(dates))).normalize().unique()
    ax.set_xticks(ticks, [date.strftime('%Y-%m-%d') for date in ticks])
    ax.grid(False)


def asset_class_weights(weights, metadata):
    columns = ['Equities', 'Bonds', 'Commodities', 'FX', 'Crypto', 'Cash']
    grouped = pd.DataFrame(0.0, index=weights.index, columns=columns)
    for ticker in weights:
        grouped[metadata.loc[ticker, 'Asset Class']] += weights[ticker]
    grouped['Cash'] = 1 - weights.sum(axis=1)
    if not np.allclose(grouped.sum(axis=1), 1, atol=1e-8) or grouped.min().min() < -1e-8:
        raise ValueError('Actual capital weights fail the 100% long-only reconciliation.')
    return grouped


def plot_comparison(returns, weights, metadata, capital, output_dir, colors,
                    class_colors, *, partial=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = returns.index[0] - pd.Timedelta(days=1)
    cumulative = pd.concat([pd.DataFrame(0.0, index=[base], columns=returns.columns),
                            (1 + returns).cumprod() - 1])
    wealth = (1 + cumulative) * capital
    suffix = ' (Partial)' if partial else ''
    period = f'{returns.index[0].date()} to {returns.index[-1].date()}'
    charts = {}
    for key, title, values, ylabel in [
        ('growth', 'Cumulative Return', cumulative, 'Cumulative Return'),
        ('wealth', 'Wealth', wealth, 'Wealth (SGD)'),
    ]:
        fig, ax = plt.subplots(figsize=(13, 6))
        for name in values:
            ax.plot(values.index, values[name], label=name, color=colors[name], linewidth=2)
        ax.set(title=f'Final Product and Benchmarks: {title}{suffix}\n{period}', xlabel='Date', ylabel=ylabel)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2) if key == 'growth' else StrMethodFormatter('{x:,.2f}'))
        _dates(ax, values.index)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
        fig.tight_layout()
        path = out / f'{key}.png'
        fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)
        charts[key] = str(path)
    drawdown = wealth / wealth.cummax() - 1
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, name in zip(axes.flat, returns.columns):
        ax.fill_between(drawdown.index, drawdown[name], 0, color=colors[name], linewidth=0)
        ax.set(title=name, xlabel='Date', ylabel='Drawdown', ylim=(min(-0.001, float(drawdown.min().min()) * 1.1), 0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
        _dates(ax, drawdown.index)
    fig.suptitle(f'Drawdown Comparison{suffix}\n{period}')
    fig.tight_layout()
    path = out / 'drawdown.png'
    fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)
    charts['drawdown'] = str(path)
    grouped = {name: asset_class_weights(frame, metadata) for name, frame in weights.items()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, (name, frame) in zip(axes.flat, grouped.items()):
        labels = list(frame.columns)
        ax.stackplot(frame.index, *[frame[c].clip(lower=0).to_numpy() for c in labels],
                     colors=[class_colors[c] for c in labels], linewidth=0)
        ax.set(title=name, xlabel='Date', ylabel='End-of-Day Capital Weight', ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
        _dates(ax, frame.index)
    handles = [Patch(facecolor=class_colors[c], label=c) for c in next(iter(grouped.values())).columns]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False)
    fig.suptitle(f'Actual Asset-Class Capital Weights{suffix}\n{period}')
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = out / 'allocation.png'
    fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)
    charts['allocation'] = str(path)
    return charts, grouped


def display_charts(charts):
    for path in charts.values():
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.imshow(plt.imread(path)); ax.axis('off'); fig.tight_layout()
    plt.show()
    plt.close('all')
