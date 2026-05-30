"""
Shared evaluation utilities used across all model notebooks.

Functions return figures or DataFrames rather than calling plt.show()
so notebooks can save them to reports/figures/ if needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       label: str = "",
                       avg_capacity: float | None = None) -> dict:
    """
    Compute RMSE, MAE and R². Optionally convert RMSE to average bike count.
    Prints a formatted summary and returns a dict.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    header = f"  {label}" if label else ""
    if header:
        print(header)
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")

    if avg_capacity:
        print(f"    RMSE in bikes : ±{rmse * avg_capacity:.1f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def naive_metrics(y_true: pd.Series,
                  y_persistence: pd.Series,
                  y_hist_mean: pd.Series) -> dict:
    """
    Evaluate two naive baselines that every ML model should beat:
      - persistence: predict current_occ (same as right now)
      - hist_mean:   predict hist_mean_occ (average for this slot)
    """
    print("=== NAIVE BASELINES ===")
    m1 = regression_metrics(y_true, y_persistence, label="Persistence (current_occ)")
    m2 = regression_metrics(y_true, y_hist_mean,   label="Historical mean (hist_mean_occ)")
    return {"persistence": m1, "hist_mean": m2}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     rmse: float,
                     r2: float,
                     title: str = "Predicted vs Actual",
                     n_sample: int = 5000,
                     seed: int = 42) -> plt.Figure:
    """Scatter plot of predicted vs actual occupancy on a random sample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(n_sample, len(y_true)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.asarray(y_true)[idx], y_pred[idx],
               alpha=0.15, s=4, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual occupancy (T+h)")
    ax.set_ylabel("Predicted occupancy (T+h)")
    ax.set_title(f"{title}\nRMSE={rmse:.4f}   R²={r2:.4f}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   title: str = "Residuals",
                   n_sample: int = 5000,
                   seed: int = 42) -> plt.Figure:
    """Histogram of residuals (actual - predicted). Should be centred on 0."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(n_sample, len(y_true)), replace=False)
    residuals = np.asarray(y_true)[idx] - y_pred[idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_error_by_group(results_df: pd.DataFrame,
                        group_col: str,
                        metric: str = "rmse",
                        title: str | None = None) -> plt.Figure:
    """
    Bar chart of a metric broken down by a categorical group (e.g. zona, hour).

    results_df must have columns: [group_col, metric].
    """
    df = results_df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df[group_col].astype(str), df[metric], color="steelblue", edgecolor="white")
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} by {group_col}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: list[str],
                            importances: np.ndarray,
                            top_n: int = 20,
                            title: str = "Feature importances") -> plt.Figure:
    """Horizontal bar chart of the top_n most important features."""
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
