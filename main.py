"""
Punto de entrada del proyecto. Orquesta el flujo completo de la fase
de modelado: carga de datos, preprocesamiento, entrenamiento con CV,
evaluación, optimización de umbral y visualización de resultados.

Uso:
    python main.py                  # carga artefactos existentes si los hay
    python main.py --force-retrain  # re-ejecuta todo ignorando artefactos
    python main.py --dataset full   # ejecuta solo sobre el dataset completo
    python main.py --dataset clean  # ejecuta solo sobre el dataset depurado
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Permite importar desde src/ sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (
    N_FOLDS,
    RESULTS_DIR,
    MODELS_DIR,
)
from data_loader import DataLoader
from evaluator import Evaluator
from persistence import Persistence
from preprocessor import Preprocessor
from trainer import Trainer
from visualizer import (
    plot_class_weights,
    plot_dataset_comparison,
    plot_fold_metrics,
    plot_pr_curves,
    plot_fold_metrics_optimal,
    plot_dataset_comparison_optimal,
)


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento y evaluación de modelos DT y XGBoost para predicción de diabetes."
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Re-ejecuta el entrenamiento completo ignorando artefactos existentes.",
    )
    parser.add_argument(
        "--dataset",
        choices=["full", "clean", "both"],
        default="both",
        help="Dataset sobre el que ejecutar el flujo (default: both).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Verificación de artefactos existentes
# ---------------------------------------------------------------------------

def artifacts_exist(dataset_tag: str) -> bool:
    """
    Verifica si los artefactos del CV y el modelo final ya existen en disco
    para el dataset indicado.

    Returns
    -------
    bool : True si el resumen del CV y los resultados finales están disponibles.
    """
    cv_summary = RESULTS_DIR / f"cv_summary_{dataset_tag}.csv"
    final_res  = RESULTS_DIR / f"final_results_{dataset_tag}.json"
    return cv_summary.exists() and final_res.exists()


# ---------------------------------------------------------------------------
# Recálculo de CV summary con umbral óptimo
# ---------------------------------------------------------------------------

def recalculate_cv_summary_with_optimal_threshold(
    trainer,
    optimal_thresholds: dict,
    dataset_tag: str,
) -> pd.DataFrame:
    """
    Recalcula el resumen del CV (summary) utilizando los umbrales óptimos
    encontrados en lugar del umbral por defecto (0.5).

    Parameters
    ----------
    trainer : Trainer
        Objeto con los fold_results (que incluyen modelos y datos).
    optimal_thresholds : dict
        Diccionario con los umbrales óptimos por modelo.
    dataset_tag : str
        Etiqueta del dataset.

    Returns
    -------
    pd.DataFrame : summary con métricas recalculadas.
    """
    import pandas as pd
    from evaluator import Evaluator
    
    rows = []
    
    # Por cada fold, evaluar los modelos con el umbral óptimo
    for result in trainer.fold_results:
        fold_id = result["fold"]
        row = {"fold": fold_id, "dataset_tag": dataset_tag}
        
        X_val_scaled = result["X_val"]
        y_val = result["y_val"]
        
        for model_name in ["dt", "xgb"]:
            if model_name not in result["models"]:
                continue
            
            model = result["models"][model_name]
            threshold = optimal_thresholds.get(model_name, 0.5)
            
            # Evaluar con el umbral óptimo
            evaluator = Evaluator(threshold=threshold)
            metrics = evaluator.evaluate(model, X_val_scaled, y_val)
            
            # Agregar métricas al row
            row[f"{model_name}_prauc"]     = metrics["prauc"]
            row[f"{model_name}_precision"] = metrics["precision"]
            row[f"{model_name}_recall"]    = metrics["recall"]
            row[f"{model_name}_f1"]        = metrics.get("f1", 0.0)
            row[f"{model_name}_threshold"] = metrics.get("threshold", threshold)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Flujo principal por dataset
# ---------------------------------------------------------------------------

def run_pipeline(dataset_tag: str) -> dict:
    """
    Ejecuta el flujo completo de entrenamiento y evaluación para un dataset.

    Pasos:
      1. Carga y partición estratificada de datos.
      2. Validación cruzada con CV de N_FOLDS folds.
      3. Guardado de métricas por fold y resumen del CV.
      4. Entrenamiento del modelo final sobre X_train completo.
      5. Evaluación sobre X_val con umbral por defecto.
      6. Optimización del umbral sobre X_tune.
      7. Evaluación final sobre X_tune con umbral optimizado.
      8. Guardado de todos los artefactos y resultados finales.

    Parameters
    ----------
    dataset_tag : str
        'full' o 'clean'.

    Returns
    -------
    dict : resultados finales listos para visualización.
    """
    use_clean = dataset_tag == "clean"
    print(f"\n{'='*60}")
    print(f"  Iniciando pipeline — dataset: {dataset_tag.upper()}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Carga y partición
    # ------------------------------------------------------------------
    print("\n[1/8] Cargando y particionando datos...")
    loader = DataLoader(use_clean=use_clean)
    loader.load().split()
    if loader.X_train is None or loader.y_train is None or loader.X_val is None or loader.y_val is None or loader.X_tune is None or loader.y_tune is None:
        raise RuntimeError("X_train / y_train / X_val / y_val / X_tune / y_tune no disponibles después de split(). Revisa DataLoader.")

    props = loader.class_proportions()
    print(f"      Proporción de clases en X_train: {props}")
    print(f"      Tamaños — train: {len(loader.X_train):,} | "
          f"val: {len(loader.X_val):,} | tune: {len(loader.X_tune):,}")

    # ------------------------------------------------------------------
    # 2. Validación cruzada
    # ------------------------------------------------------------------
    print(f"\n[2/8] Ejecutando CV estratificado ({N_FOLDS} folds)...")
    trainer = Trainer(loader=loader, dataset_tag=dataset_tag)
    trainer.run_cv()

    # ------------------------------------------------------------------
    # 3. Guardado de métricas por fold y resumen del CV
    # ------------------------------------------------------------------
    print("\n[3/8] Guardando métricas por fold...")
    for result in trainer.fold_results:
        path = Persistence.save_fold_metrics(
            metrics     = result["metrics"],
            fold_id     = result["fold"],
            dataset_tag = dataset_tag,
        )
        for model_name, model in result["models"].items():
            Persistence.save_model(
                model       = model,
                model_name  = model_name,
                fold_id     = result["fold"],
                dataset_tag = dataset_tag,
            )
        print(f"      Fold {result['fold']} guardado → {path.name}")

    summary = trainer.get_cv_summary()
    summary_path = Persistence.save_cv_summary(summary, dataset_tag)
    print(f"      Resumen del CV guardado → {summary_path.name}")
    print(f"\n      Promedios del CV:")
    for col in summary.columns:
        if col not in ("fold", "dataset_tag"):
            print(f"        {col}: {summary[col].mean():.4f} ± {summary[col].std():.4f}")

    # ------------------------------------------------------------------
    # 4. Entrenamiento del modelo final
    # ------------------------------------------------------------------
    print("\n[4/8] Entrenando modelos finales sobre X_train completo...")
    trainer.train_final()
    for model_name, model in trainer.final_models.items():
        if model_name == "preprocessor":
            continue
        Persistence.save_model(
            model       = model,
            model_name  = model_name,
            fold_id     = "final",
            dataset_tag = dataset_tag,
        )
    print("      Modelos finales guardados.")

    # ------------------------------------------------------------------
    # 5. Evaluación sobre X_val con umbral por defecto
    # ------------------------------------------------------------------
    print("\n[5/8] Evaluando sobre X_val (umbral por defecto)...")
    prep_final = trainer.final_models["preprocessor"]
    X_val_scaled  = prep_final.transform(loader.X_val)
    X_tune_scaled = prep_final.transform(loader.X_tune)

    evaluator = Evaluator()
    val_metrics = {
        model_name: evaluator.evaluate(model, X_val_scaled, loader.y_val)
        for model_name, model in trainer.final_models.items()
        if model_name not in ("preprocessor", "weights")
    }
    print("      Métricas sobre X_val:")
    for model_name, metrics in val_metrics.items():
        print(f"        {model_name.upper()}: PR-AUC={metrics['prauc']:.4f} | "
              f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}")

    # ------------------------------------------------------------------
    # 6. Optimización del umbral sobre X_tune
    # ------------------------------------------------------------------
    print("\n[6/8] Optimizando umbral sobre X_tune...")
    optimal_thresholds = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator()
        threshold = opt_eval.optimize_threshold(
            model      = model,
            X          = X_tune_scaled,
            y          = loader.y_tune,
            strategy   = "prauc",
        )
        optimal_thresholds[model_name] = threshold
        print(f"      {model_name.upper()} → umbral óptimo: {threshold:.2f}")

    # ------------------------------------------------------------------
    # 7. Evaluación final con umbral optimizado
    # ------------------------------------------------------------------
    print("\n[7.1/8] Evaluando sobre X_tune con umbral optimizado...")
    tune_metrics = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        tune_metrics[model_name] = opt_eval.evaluate(model, X_tune_scaled, loader.y_tune)

    print("      Métricas sobre X_tune (umbral optimizado):")
    for model_name, metrics in tune_metrics.items():
        print(f"        {model_name.upper()}: PR-AUC={metrics['prauc']:.4f} | "
              f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
              f"Umbral={metrics['threshold']:.2f}")
        
    print("\n[7.2/8] Evaluando sobre X_val con umbral optimizado...")
    tune_metrics = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        tune_metrics[model_name] = opt_eval.evaluate(model, X_val_scaled, loader.y_val)

    print("      Métricas sobre X_val (umbral optimizado):")
    for model_name, metrics in tune_metrics.items():
        print(f"        {model_name.upper()}: PR-AUC={metrics['prauc']:.4f} | "
              f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
              f"Umbral={metrics['threshold']:.2f}")

    # ------------------------------------------------------------------
    # 8. Guardado de resultados finales
    # ------------------------------------------------------------------
    print("\n[8/8] Guardando resultados finales...")
    final_results = {
        "dataset_tag"       : dataset_tag,
        "optimal_thresholds": optimal_thresholds,
        "val_metrics"       : val_metrics,
        "tune_metrics"      : tune_metrics,
        "class_weights"     : {
            str(k): v for k, v in trainer.final_models["weights"].items()
        },
    }
    results_path = Persistence.save_final_results(final_results, dataset_tag)
    print(f"      Resultados guardados → {results_path.name}")

    # ------------------------------------------------------------------
    # Recálculo de CV summary con umbral óptimo (para visualizaciones)
    # ------------------------------------------------------------------
    print("\n[Bonus/8] Recalculando CV summary con umbral óptimo...")
    summary_optimal = recalculate_cv_summary_with_optimal_threshold(
        trainer,
        optimal_thresholds,
        dataset_tag,
    )
    print(f"      Summary recalculado con umbral óptimo.")

    return {
        "summary"         : summary,
        "summary_optimal" : summary_optimal,
        "final_results"   : final_results,
        "trainer"         : trainer,
        "loader"          : loader,
        "prep_final"      : prep_final,
    }


# ---------------------------------------------------------------------------
# Carga de resultados existentes
# ---------------------------------------------------------------------------

def load_pipeline_results(dataset_tag: str) -> dict:
    """
    Recupera los artefactos de una ejecución previa desde disco.

    Returns
    -------
    dict con 'summary' y 'final_results'.
    """
    print(f"\n[INFO] Artefactos encontrados para '{dataset_tag}'. Cargando desde disco...")
    return {
        "summary"      : Persistence.load_cv_summary(dataset_tag),
        "final_results": Persistence.load_final_results(dataset_tag),
        "trainer"      : None,
        "loader"       : None,
        "prep_final"   : None,
    }


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def run_visualizations(results: dict, dataset_tag: str) -> None:
    """
    Genera y guarda todas las visualizaciones para un dataset.

    Requiere que el pipeline haya sido ejecutado (no solo cargado desde disco),
    ya que algunas visualizaciones necesitan acceso al loader y al trainer.

    Parameters
    ----------
    results : dict
        Resultado de run_pipeline().
    dataset_tag : str
        Etiqueta del dataset.
    """
    summary = results["summary"]
    summary_optimal = results.get("summary_optimal")
    trainer = results["trainer"]
    loader  = results["loader"]
    prep    = results["prep_final"]

    # métricas por fold
    path = plot_fold_metrics(summary, dataset_tag)
    print(f"      [Fig] Métricas por fold → {path.name}")

    # métricas por fold (umbral óptimo) — solo si está disponible
    if summary_optimal is not None:
        path = plot_fold_metrics_optimal(summary_optimal, dataset_tag)
        print(f"      [Fig] Métricas por fold → {path.name}")

    # pesos de clase por fold (solo si el trainer está disponible)
    if trainer is not None:
        path = plot_class_weights(trainer.fold_results, dataset_tag)
        print(f"      [Fig] Pesos de clase → {path.name}")

    # curvas PR sobre X_val (solo si el pipeline fue ejecutado)
    if loader is not None and trainer is not None and prep is not None:
        evaluator = Evaluator()
        X_val_scaled = prep.transform(loader.X_val)
        curve_data = {
            model_name: evaluator.evaluate_curve(model, X_val_scaled, loader.y_val)
            for model_name, model in trainer.final_models.items()
            if model_name not in ("preprocessor", "weights")
        }
        path = plot_pr_curves(curve_data, dataset_tag)
        print(f"      [Fig] Curvas PR → {path.name}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args     = parse_args()
    tags     = ["full", "clean"] if args.dataset == "both" else [args.dataset]
    all_data = {}

    print("\n=== Diabetes XAI Pipeline ===")

    # ejecutar o cargar pipeline por dataset
    for tag in tags:
        if not args.force_retrain and artifacts_exist(tag):
            all_data[tag] = load_pipeline_results(tag)
        else:
            all_data[tag] = run_pipeline(tag)

    # visualizaciones individuales por dataset
    print("\n--- Generando visualizaciones ---")
    for tag in tags:
        run_visualizations(all_data[tag], tag)

    # comparación entre datasets (solo si se ejecutaron ambos)
    if len(all_data) == 2 and all(
        all_data[t]["summary"] is not None for t in ["full", "clean"]
    ):
        path = plot_dataset_comparison(
            all_data["full"]["summary"],
            all_data["clean"]["summary"],
        )
        print(f"      [Fig] Comparación full vs. clean → {path.name}")

        # Comparación con umbral óptimo (solo si ambos tienen summary_optimal)
        if all(
            all_data[t].get("summary_optimal") is not None for t in ["full", "clean"]
        ):
            path = plot_dataset_comparison_optimal(
                all_data["full"]["summary_optimal"],
                all_data["clean"]["summary_optimal"],
            )
            print(f"      [Fig] Comparación full vs. clean → {path.name}")

    print("\n✓ Pipeline completado.\n")