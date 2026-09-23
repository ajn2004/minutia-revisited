"""Plots generated from saved reproduction results."""

import csv
from pathlib import Path


def plot_learning_curves(csv_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(csv_path).open(newline="")))
    names = sorted({row["regime"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        subset = [row for row in rows if row["regime"] == name]
        x = [int(row["iteration"]) for row in subset]
        ax.plot(
            x,
            [float(row["detection_efficiency"]) for row in subset],
            label="detection efficiency / recall",
        )
        ax.plot(
            x,
            [float(row["false_identification_fraction"]) for row in subset],
            label="false-identification fraction",
        )
        ax.set_title(name)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
