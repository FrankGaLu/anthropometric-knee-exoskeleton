"""Generate summary plots from aggregate CSVs; no source archive is required."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    table = pd.read_csv(ROOT / 'results/model_metrics.csv')
    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout='constrained')
    x = np.arange(len(table))
    for offset, field, label, color in [(-0.19, 'mae_mm', 'Test MAE', '#16324f'),
                                        (0.19, 'rmse_mm', 'Test RMSE', '#a9bba0')]:
        bars = ax.bar(x + offset, table[field], 0.35, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=table.model, ylabel='Error (mm)', ylim=(0, 34))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output / 'model_summary.pdf')
    plt.close(fig)
    print('Generated figures/model_summary.pdf from aggregate exports; models were not refitted.')


if __name__ == '__main__':
    main()
