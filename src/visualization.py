"""Layer 7 — Visualization helpers (matplotlib, no seaborn dependency)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_df: pd.DataFrame, save_path: str | None = None):
    """Line plot of predicted label id over twin history."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if history_df.empty or "pred_id" not in history_df:
        ax.text(0.5, 0.5, "No history yet", ha="center")
    else:
        ax.plot(range(len(history_df)), history_df["pred_id"].fillna(0),
                marker="o")
        ax.set_xlabel("Update #")
        ax.set_ylabel("Predicted class id")
        ax.set_title("Digital-twin risk trajectory")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_top3(risk: dict, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    top3 = risk.get("top3", [])
    ax.barh([t["type"] for t in top3][::-1],
            [t["prob"] for t in top3][::-1])
    ax.set_xlabel("Probability")
    ax.set_title(f"Top-3 types (score {risk.get('score')}, {risk.get('level')})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_model_comparison(results: dict, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(results.keys())
    f1s = [results[k]["f1_macro"] for k in names]
    ax.bar(names, f1s)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
