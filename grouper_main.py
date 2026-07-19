"""
Punto de entrada para la etapa de Agrupación por Probabilidades (Grouping).
Permite graficar la distribución de cualquier modelo y extraer vecindades locales (anclas).
"""
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_logger, CONTINUOUS_COLS, ORDINAL_COLS, RESULTS_GROUP_DIR, FIGURES_GROUP_DIR
from src.binary_clasification.data_loader import DataLoader
from src.binary_clasification.evaluator import Evaluator
from src.binary_clasification.persistence import Persistence
from src.binary_clasification.preprocessor import Preprocessor
from src.grouper.probability_grouper import ProbabilityGrouper

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Módulo de Agrupación por Probabilidades (Grouping).")
    
    parser.add_argument("--model", type=str, default="xgb", help="Nombre del modelo pre-entrenado a cargar (ej. dt, xgb).")
    parser.add_argument("--dataset", choices=["full", "clean"], default="clean", help="Qué variante del dataset usar.")
    parser.add_argument("--partition", choices=["train", "val", "tune", "all"], default="val", help="Qué partición de datos analizar.")
    
    parser.add_argument("--plot-proba", action="store_true", default=False, help="Graficar histograma de probabilidades.")
    parser.add_argument("--group-analysis", action="store_true", default=False, help="Ejecutar partición y selección de vecindad.")
    
    parser.add_argument("--group-min", type=float, default=0.0, help="Límite inferior de probabilidad del subgrupo.")
    parser.add_argument("--group-max", type=float, default=0.25, help="Límite superior de probabilidad del subgrupo.")
    parser.add_argument("--anchor-prob", type=float, default=None, help="Probabilidad objetivo para el ancla (opcional).")
    parser.add_argument("--radius", type=float, default=0.3, help="Radio máximo de distancia de Gower para la vecindad.")
    
    return parser.parse_args()


def load_partition_data(loader: DataLoader, partition: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extrae la partición de datos solicitada."""
    loader = loader.load().split()
    
    if partition == "train":
        if loader.X_train is None or loader.y_train is None:
            logger.error("Partición de entrenamiento no disponible. Asegúrate de que el dataset esté cargado y dividido correctamente.")
            raise ValueError("Partición de entrenamiento no disponible.")
        return loader.X_train, loader.y_train
    elif partition == "val":
        if loader.X_val is None or loader.y_val is None:
            logger.error("Partición de validación no disponible. Asegúrate de que el dataset esté cargado y dividido correctamente.")
            raise ValueError("Partición de validación no disponible.")
        return loader.X_val, loader.y_val
    elif partition == "tune":
        if loader.X_tune is None or loader.y_tune is None:
            logger.error("Partición de ajuste (tuning) no disponible. Asegúrate de que el dataset esté cargado y dividido correctamente.")
            raise ValueError("Partición de ajuste (tuning) no disponible.")
        return loader.X_tune, loader.y_tune
    else:  # all
        X_all = pd.concat([loader.X_train, loader.X_val, loader.X_tune]).reset_index(drop=True)
        y_all = pd.concat([loader.y_train, loader.y_val, loader.y_tune]).reset_index(drop=True)
        return X_all, y_all


def plot_proba_histogram(probas: np.ndarray, model_tag: str, context_tag: str) -> None:
    """Genera y guarda el histograma de probabilidades en la carpeta dedicada de agrupación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probas, bins=50, color="#0FA6E7", edgecolor="black", alpha=0.7)
    ax.set_title(f"Distribución de Probabilidades - {model_tag.upper()} ({context_tag})")
    ax.set_xlabel("Probabilidad Predicha")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    filename = f"proba_hist_{model_tag}_{context_tag}.png"
    path = FIGURES_GROUP_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Histograma de probabilidades guardado en: {path.name}")


def main():
    args = parse_args()
    logger.info(f"=== Iniciando Módulo de Agrupación (Modelo: {args.model.upper()} | Partición: {args.partition}) ===")
    
    loader = DataLoader(use_clean=(args.dataset == "clean"))
    X_data, _ = load_partition_data(loader, args.partition)
    
    try:
        model = Persistence.load_model(args.model, fold_id=0, dataset_tag=args.dataset)
    except FileNotFoundError:
        logger.error(f"No se encontró el modelo {args.model} para el dataset {args.dataset}. Ejecute clf_main.py primero.")
        sys.exit(1)
        
    try:
        preprocessor = Preprocessor.load(fold_id=0, dataset_tag=args.dataset)
        X_scaled = X_data.copy()
        X_scaled[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(X_data[CONTINUOUS_COLS])
        logger.debug("Datos escalados utilizando el preprocesador pre-entrenado.")
    except FileNotFoundError:
        X_scaled = X_data
        logger.warning("No se encontró preprocesador, procediendo con datos sin escalar (puede afectar predicciones).")

    evaluator = Evaluator()
    probas = evaluator._get_proba(model, X_scaled)
    context_tag = f"{args.dataset}_{args.partition}"
    
    if args.plot_proba:
        plot_proba_histogram(probas, args.model, context_tag)
        
    if args.group_analysis:
        logger.info(f"Iniciando segmentación: Intervalo [{args.group_min}, {args.group_max}], Ancla: {args.anchor_prob or 'No especificado'}, Radio: {args.radius}")
        
        grouper = ProbabilityGrouper(X_data, probas)
        
        subgroup = grouper.get_subgroup(args.group_min, args.group_max)

        if args.anchor_prob is not None:
            anchor = grouper.get_anchor(subgroup, args.anchor_prob, args.group_min, args.group_max)
            feature_cols = CONTINUOUS_COLS + ORDINAL_COLS
            neighborhood = grouper.get_neighborhood(subgroup, anchor, args.radius, feature_cols)
            out_file = RESULTS_GROUP_DIR / f"neighborhood_{args.model}_{context_tag}_anchor{args.anchor_prob}_radius_{args.radius}.csv"
            neighborhood.to_csv(out_file, index=False)
            logger.info(f"Análisis de agrupamiento finalizado. Guardado en: {out_file.name}")
        else:
            logger.info("No se proporcionó --anchor-prob; se omitió la generación de vecindad basada en ancla.")

    if not args.plot_proba and not args.group_analysis:
        logger.warning("No se especificó ninguna tarea. Use --plot-proba o --group-analysis.")

if __name__ == "__main__":
    main()