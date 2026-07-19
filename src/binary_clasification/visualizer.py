import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import FIGURES_CLF_DIR, MODEL_CONFIGS, get_logger

logger = get_logger(__name__)

def _save_fig(fig: Figure, filename: str) -> Path:
    path = FIGURES_CLF_DIR / filename
    
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.debug(f"Figura guardada exitosamente: {filename}")
    except Exception as e:
        logger.error(f"Fallo al guardar la figura {filename}: {e}")
        raise
    finally:
        plt.close(fig)
        
    return path

def _model_colors() -> dict:
    return {tag: cfg["color"] for tag, cfg in MODEL_CONFIGS.items()}

def _model_display_name(model_tag: str) -> str:
    return MODEL_CONFIGS.get(model_tag, {}).get("display_name", model_tag.upper())

def plot_fold_metrics(summary: pd.DataFrame, dataset_tag: str, metrics: list | None = None, is_optimal: bool = False, thresh_metric: str | None = None) -> Path:
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]

    models = list(MODEL_CONFIGS.keys())
    colors = _model_colors()
    n_folds = len(summary)
    x = np.arange(n_folds)
    width = 0.35 / max(1, len(models) / 2)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            col = f"{model}_{metric}"
            values = summary[col].values if col in summary.columns else np.zeros(n_folds)
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors.get(model))
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_title(metric.upper().replace("PRAUC", "PR-AUC"), fontsize=11)
        ax.set_xlabel("Fold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in summary["fold"].values])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    suffix_title = f" (Óptimo, estrategia={thresh_metric})" if is_optimal and thresh_metric else (" (Óptimo)" if is_optimal else "")
    suffix_file = f"_optimal_{thresh_metric}" if is_optimal and thresh_metric else ("_optimal" if is_optimal else "")

    fig.suptitle(f"Métricas por fold{suffix_title} — dataset: {dataset_tag}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"fold_metrics_{dataset_tag}{suffix_file}.png")

def plot_pr_curves(curve_data: dict, dataset_tag: str) -> Path:
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
    ax.set_title(f"Curvas Precision-Recall — dataset: {dataset_tag}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_fig(fig, f"pr_curves_{dataset_tag}.png")

def plot_dataset_comparison(summary_full: pd.DataFrame, summary_clean: pd.DataFrame, metrics: list | None = None, is_optimal: bool = False, thresh_metric: str | None = None) -> Path:
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]

    models = list(MODEL_CONFIGS.keys())
    datasets = {"full": summary_full, "clean": summary_clean}
    colors = {"full": "#4C72B0", "clean": "#55A868"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(models))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, (tag, summary) in enumerate(datasets.items()):
            means = [summary[f"{m}_{metric}"].mean() for m in models]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width, label=f"Dataset: {tag}", color=colors[tag])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_title(metric.upper().replace("PRAUC", "PR-AUC"), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    suffix_title = f" (Óptimo, estrategia={thresh_metric})" if is_optimal and thresh_metric else (" (Óptimo)" if is_optimal else "")
    suffix_file = f"_optimal_{thresh_metric}" if is_optimal and thresh_metric else ("_optimal" if is_optimal else "")

    fig.suptitle(f"Comparación de métricas: full vs. clean{suffix_title}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"dataset_comparison{suffix_file}.png")

def plot_class_weights(fold_results: list, dataset_tag: str) -> Path:
    folds = [r["fold"] for r in fold_results]
    w_neg = [r["weights"][0] for r in fold_results]
    w_pos = [r["weights"][1] for r in fold_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(folds, w_neg, marker="o", label="Clase 0 (sin diabetes)", color="#4C72B0")
    ax.plot(folds, w_pos, marker="s", label="Clase 1 (diabetes)", color="#DD8452")

    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Peso asignado", fontsize=11)
    ax.set_xticks(folds)
    ax.set_title(f"Pesos de clase por fold — dataset: {dataset_tag}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_fig(fig, f"class_weights_{dataset_tag}.png")

def plot_final_metrics(final_results: dict, dataset_tag: str, metrics: list | None = None) -> Path | None:
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]
        
    val_metrics = final_results.get("val_metrics_opt", final_results.get("val_metrics", {}))
    models = list(val_metrics.keys())
    if not models:
        return None

    thresh_metric = final_results.get("thresh_metric", "na")

    x = np.arange(len(metrics))
    width = 0.35
    colors = _model_colors()

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, model in enumerate(models):
        values = [val_metrics[model].get(m, 0) for m in metrics]
        offset = (i - 0.5 * (len(models) - 1)) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors.get(model, "#333"))
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_title(f"Evaluación Final (Test 20%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return _save_fig(fig, f"final_metrics_{dataset_tag}_{thresh_metric}.png")

def plot_final_comparison(res_full: dict, res_clean: dict, metrics: list | None = None) -> Path | None:
    if metrics is None:
        metrics = ["prauc", "precision", "recall"]
        
    val_full = res_full.get("val_metrics_opt", res_full.get("val_metrics", {}))
    val_clean = res_clean.get("val_metrics_opt", res_clean.get("val_metrics", {}))

    models = list(set(list(val_full.keys()) + list(val_clean.keys())))
    if not models:
        return None

    thresh_metric_full = res_full.get("thresh_metric", "na")
    thresh_metric_clean = res_clean.get("thresh_metric", "na")
    if thresh_metric_full != thresh_metric_clean:
        logger.warning(
            f"plot_final_comparison recibió resultados con distinta estrategia de umbral "
            f"(full='{thresh_metric_full}' vs clean='{thresh_metric_clean}'); la comparación "
            "sigue siendo válida por modelo, pero no es un experimento equivalente entre datasets."
        )
    thresh_metric = thresh_metric_full if thresh_metric_full == thresh_metric_clean else f"{thresh_metric_full}-vs-{thresh_metric_clean}"

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(models))
    width = 0.35
    colors = {"full": "#4C72B0", "clean": "#55A868"}

    for ax, metric in zip(axes, metrics):
        means_full = [val_full.get(m, {}).get(metric, 0) for m in models]
        means_clean = [val_clean.get(m, {}).get(metric, 0) for m in models]

        b1 = ax.bar(x - width/2, means_full, width, label="Dataset: full", color=colors["full"])
        b2 = ax.bar(x + width/2, means_clean, width, label="Dataset: clean", color=colors["clean"])

        ax.bar_label(b1, fmt="%.3f", fontsize=7)
        ax.bar_label(b2, fmt="%.3f", fontsize=7)

        ax.set_title(metric.upper().replace("PRAUC", "PR-AUC"), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Comparación de Evaluación Final (Test 20%) — umbral: {thresh_metric}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"final_comparison_{thresh_metric}.png")

def plot_optuna_history(csv_path: Path, model_tag: str, dataset_tag: str, metric: str = "PR-AUC") -> Path | None:
    if not csv_path.exists():
        logger.warning(f"Historial no encontrado: {csv_path.name}")
        return None

    df = pd.read_csv(csv_path)
    
    status_col = "state" if "state" in df.columns else "status"
    if status_col in df.columns:
        df = df[df[status_col] == "COMPLETE"].copy()

    if df.empty:
        logger.warning(f"El historial en {csv_path.name} está vacío tras filtrar ensayos completos.")
        return None

    iterations = df["number"] + 1
    values = df["value"]

    best_so_far = values.cummax()
    best_idx = values.idxmax()
    best_iter = iterations.loc[best_idx]
    best_val = values.loc[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(iterations, values, alpha=0.4, color="#0FA6E7", s=15, label="Ensayo individual")

    color_line = "#DD8452" if model_tag == "xgb" else "#4C72B0"
    ax.plot(iterations, best_so_far, drawstyle="steps-post", color=color_line, linewidth=2, label=f"Mejor {metric} acumulado")

    ax.scatter([best_iter], [best_val], color="red", s=50, zorder=5, label=f"Óptimo ({best_val:.4f} en iter {best_iter})")

    model_name = _model_display_name(model_tag)
    ds_name = "completo" if dataset_tag == "full" else "depurado"

    ax.set_title(f"Historial de {metric} en {model_name} sobre dataset {ds_name} usando Optuna", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteración", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    return _save_fig(fig, f"{model_tag}_optuna_history_{dataset_tag}.png")

def plot_proba_histogram(probas: np.ndarray, model_tag: str, dataset_tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(probas, bins=50, color="#0FA6E7", edgecolor="black", alpha=0.7)
    ax.set_title(f"Distribución de Probabilidades - {model_tag.upper()} ({probas.size} muestras)")
    ax.set_xlabel("Probabilidad Predicha")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    filename = f"proba_hist_{model_tag}_{dataset_tag}_val.png"
    return _save_fig(fig, filename)

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_tag: str, dataset_tag: str, partition: str, threshold: float) -> Path:
    """
    Genera y guarda la matriz de confusión sobre una partición específica,
    dejando explícito en título y nombre de archivo qué umbral de decisión
    se usó para convertir probabilidades en clases.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negativo (0)", "Positivo (1)"], 
                yticklabels=["Negativo (0)", "Positivo (1)"], ax=ax)
    
    model_name = _model_display_name(model_tag)
    ax.set_title(
        f"Matriz de Confusión - {model_name}\n",
        #f"Dataset: {dataset_tag} | Partición: {partition} | Umbral: {threshold:.3f}",
        #f"Umbral: {threshold:.3f}",
        fontsize=11, fontweight="bold"
    )
    
    ax.set_xlabel("Predicción del Modelo", fontsize=10)
    ax.set_ylabel("Valor Real", fontsize=10)
    
    fig.tight_layout()
    
    filename = f"{model_tag}_confusion_matrix_{dataset_tag}_{partition}_t{threshold:.3f}.png"
    return _save_fig(fig, filename)