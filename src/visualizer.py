"""
Generación y guardado de visualizaciones para la fase de modelado.

Todas las funciones guardan la figura en FIGURES_DIR y devuelven la ruta.
Ninguna llama a plt.show(); el flujo es completamente no interactivo.

Visualizaciones cubiertas en este módulo:
  1. Métricas por fold (barras agrupadas por modelo).
  2. Curvas Precision-Recall por modelo y dataset.
  3. Comparación de métricas entre datasets 'full' y 'clean'.
  4. Pesos de clase por fold.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path

from config import FIGURES_DIR

# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _save_fig(fig: Figure, filename: str) -> Path:
    """Guarda la figura en FIGURES_DIR y cierra el objeto para liberar memoria."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _model_colors() -> dict:
    """Paleta de colores consistente para DT y XGBoost en todas las figuras."""
    return {"dt": "#4C72B0", "xgb": "#DD8452"}


# ---------------------------------------------------------------------------
# 1. Métricas por fold
# ---------------------------------------------------------------------------

def plot_fold_metrics(
    summary: pd.DataFrame,
    dataset_tag: str,
    metrics: list | None = None,
) -> Path:
    """
    Genera un gráfico de barras agrupadas con las métricas de cada fold
    para ambos modelos.

    Parameters
    ----------
    summary : pd.DataFrame
        Resultado de Trainer.get_cv_summary(). Debe contener columnas con
        el patrón {modelo}_{métrica} (ej. dt_prauc, xgb_recall).
    dataset_tag : str
        Etiqueta del dataset ('full' o 'clean'). Se usa en el título y
        en el nombre del archivo.
    metrics : list of str, optional
        Métricas a graficar. Por defecto: ['prauc', 'precision', 'recall'].

    Returns
    -------
    Path : ruta de la figura guardada.
    """
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]

    models  = ["dt", "xgb"]
    colors  = _model_colors()
    n_folds = len(summary)
    x       = np.arange(n_folds)
    width   = 0.35

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(5 * len(metrics), 4),
        sharey=False,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            col    = f"{model}_{metric}"
            values = summary[col].values if col in summary.columns else np.zeros(n_folds)
            offset = (i - 0.5) * width
            bars   = ax.bar(x + offset, values, width, label=model.upper(), color=colors[model])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_title(metric.upper().replace("PRAUC", "PR-AUC"), fontsize=11)
        ax.set_xlabel("Fold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in summary["fold"].values])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Métricas por fold — dataset: {dataset_tag}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return _save_fig(fig, f"fold_metrics_{dataset_tag}.png")


# ---------------------------------------------------------------------------
# 2. Curvas Precision-Recall
# ---------------------------------------------------------------------------

def plot_pr_curves(
    curve_data: dict,
    dataset_tag: str,
) -> Path:
    """
    Grafica las curvas Precision-Recall para DT y XGBoost sobre un mismo eje.

    Parameters
    ----------
    curve_data : dict
        Estructura esperada:
        {
          'dt' : {'precision_curve': array, 'recall_curve': array, 'prauc': float},
          'xgb': {'precision_curve': array, 'recall_curve': array, 'prauc': float},
        }
        Resultado directo de Evaluator.evaluate_curve() para cada modelo.
    dataset_tag : str
        Etiqueta del dataset.

    Returns
    -------
    Path : ruta de la figura guardada.
    """
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(6, 5))

    for model, data in curve_data.items():
        label = f"{model.upper()} (PR-AUC = {data['prauc']:.3f})"
        ax.plot(
            data["recall_curve"],
            data["precision_curve"],
            label=label,
            color=colors.get(model, None),
            linewidth=1.8,
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Curvas Precision-Recall — dataset: {dataset_tag}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_fig(fig, f"pr_curves_{dataset_tag}.png")


# ---------------------------------------------------------------------------
# 3. Comparación entre datasets full y clean
# ---------------------------------------------------------------------------

def plot_dataset_comparison(
    summary_full: pd.DataFrame,
    summary_clean: pd.DataFrame,
    metrics: list | None = None,
) -> Path:
    """
    Compara las métricas promedio del CV entre los datasets 'full' y 'clean'
    para ambos modelos mediante barras agrupadas.

    Parameters
    ----------
    summary_full : pd.DataFrame
        Resumen del CV sobre el dataset completo.
    summary_clean : pd.DataFrame
        Resumen del CV sobre el dataset sin outliers.
    metrics : list of str, optional
        Métricas a comparar. Por defecto: ['prauc', 'precision', 'recall'].

    Returns
    -------
    Path : ruta de la figura guardada.
    """
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]

    models   = ["dt", "xgb"]
    datasets = {"full": summary_full, "clean": summary_clean}
    colors   = {"full": "#4C72B0", "clean": "#55A868"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    x     = np.arange(len(models))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, (tag, summary) in enumerate(datasets.items()):
            means  = [summary[f"{m}_{metric}"].mean() for m in models]
            offset = (i - 0.5) * width
            bars   = ax.bar(
                x + offset, means, width,
                label=f"Dataset: {tag}",
                color=colors[tag],
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_title(metric.upper().replace("PRAUC", "PR-AUC"), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Comparación de métricas: full vs. clean (promedio CV)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return _save_fig(fig, "dataset_comparison.png")


# ---------------------------------------------------------------------------
# 4. Pesos de clase por fold
# ---------------------------------------------------------------------------

def plot_class_weights(
    fold_results: list,
    dataset_tag: str,
) -> Path:
    """
    Visualiza la evolución de los pesos de clase a lo largo de los folds.

    Parameters
    ----------
    fold_results : list of dict
        Lista self.fold_results de Trainer. Cada elemento debe contener
        la clave 'weights' con un dict {0: float, 1: float}.
    dataset_tag : str
        Etiqueta del dataset.

    Returns
    -------
    Path : ruta de la figura guardada.
    """
    folds    = [r["fold"] for r in fold_results]
    w_neg    = [r["weights"][0] for r in fold_results]
    w_pos    = [r["weights"][1] for r in fold_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(folds, w_neg, marker="o", label="Clase 0 (sin diabetes)", color="#4C72B0")
    ax.plot(folds, w_pos, marker="s", label="Clase 1 (diabetes)",     color="#DD8452")

    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Peso asignado", fontsize=11)
    ax.set_xticks(folds)
    ax.set_title(
        f"Pesos de clase por fold — dataset: {dataset_tag}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_fig(fig, f"class_weights_{dataset_tag}.png")