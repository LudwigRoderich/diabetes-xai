import argparse
import sys
import json
import pandas as pd


from src.config import (
    N_FOLDS, OPTUNA_TRIALS, OPTUNA_PATIENCE, OPTUNA_TOLERANCE,
    MODEL_CONFIGS, MODEL_TAGS, DATASET_TAGS, PARTITION_TAGS, OPTIMIZATION_METRIC_CHOICES, THRESH_METRIC_CHOICES,
    get_logger, RESULTS_CLF_DIR
)

from src.binary_clasification.data_loader import DataLoader
from src.binary_clasification.evaluator import Evaluator
from src.binary_clasification.persistence import Persistence
from src.binary_clasification.trainer import Trainer
from src.binary_clasification.optimizer import HyperparameterOptimizer
from src.binary_clasification.preprocessor import Preprocessor
from src.binary_clasification.inference import ModelInference
from src.binary_clasification.visualizer import (
    plot_class_weights, plot_dataset_comparison, plot_fold_metrics,
    plot_pr_curves, plot_final_metrics, plot_final_comparison, plot_optuna_history,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de Clasificación Binaria.")
    parser.add_argument("--force-retrain", action="store_true", default=False)
    parser.add_argument("--dataset", choices=[*DATASET_TAGS, "all"], default="all")
    parser.add_argument("--optimize", action="store_true", default=False)
    parser.add_argument("--model", choices=[*MODEL_TAGS, "all"], default="all")
    parser.add_argument(
        "--optimize-metric", choices=list(OPTIMIZATION_METRIC_CHOICES), default="prauc",
        help="Métrica que Optuna maximiza durante la búsqueda de hiperparámetros (--optimize). "
             "No confundir con --thresh-metric: esa elige la ESTRATEGIA de umbral de decisión "
             "sobre un modelo ya entrenado; esta elige el OBJETIVO de la búsqueda de hiperparámetros."
    )
    parser.add_argument("--partition", choices=list(PARTITION_TAGS), default="val", help="Partición a usar")
    parser.add_argument("--thresh-metric", choices=THRESH_METRIC_CHOICES, default="f1")
    parser.add_argument("--skip-cv", action="store_true", default=False)
    parser.add_argument("--plot-history", action="store_true", default=False)
    parser.add_argument("--plot-cm", action="store_true", default=False, help="Habilita la gráfica de la matriz de confusión")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Umbral manual para --plot-cm (override de pruebas; ignora el umbral óptimo persistido)."
    )
    return parser.parse_args()

import sys

def validate_context_args(args, expected_args: list[str], context_name: str):
    """
    Compara los argumentos pasados explícitamente en consola contra 
    los argumentos permitidos para un bloque de ejecución.
    Muestra advertencias por parámetros inútiles y notifica los valores por defecto usados.
    """
    passed_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            raw_arg = arg.lstrip("-").split("=")[0]
            passed_args.add(raw_arg.replace("-", "_"))

    expected_set = set(expected_args)
    
    ignored = passed_args - expected_set
    if ignored:
        logger.warning(f"[{context_name}] Ignorando parámetros que no aplican: {', '.join(ignored)}")
        
    defaults_used = expected_set - passed_args
    if defaults_used:
        defaults_info = []
        for d in defaults_used:
            if hasattr(args, d):
                val = getattr(args, d)
                defaults_info.append(f"{d}={val}")
        if defaults_info:
            logger.info(f"[{context_name}] Usando valores por defecto para: {', '.join(defaults_info)}")


def artifacts_exist(dataset_tag: str, thresh_metric: str) -> bool:
    """
    Verifica si ya existen modelos y el experimento (dataset + estrategia de
    umbral) específico entrenados para el pipeline de clasificación.
    """
    models = Persistence.list_saved_models(dataset_tag)
    if not models:
        return False
    results_path = RESULTS_CLF_DIR / f"final_results_{dataset_tag}_{thresh_metric}.json"
    return results_path.exists()


def load_optimized_params(model_name: str, dataset_tag: str) -> dict | None:
    filepath = RESULTS_CLF_DIR / f"{model_name}_optimization_best_params_{dataset_tag}.json"
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Parámetros óptimos cargados para {model_name.upper()} ({dataset_tag})")
                return data.get("best_params", None)
        except Exception as e:
            logger.error(f"Error al leer parámetros en {filepath.name}: {e}")
    return None


def build_model_params(target_model: str, dataset_tag: str) -> dict[str, dict]:
    """
    Construye, para cada modelo objetivo, sus hiperparámetros finales:
    base_params (config.py) sobreescritos por los optimizados (Optuna) si
    existen, fusionados con los fixed_params del modelo (los que no se
    tunean, ej. device/tree_method en XGBoost). Generalizado sobre
    MODEL_CONFIGS: agregar un modelo nuevo no requiere tocar esta función.
    """
    model_params: dict[str, dict] = {}
    for tag, cfg in MODEL_CONFIGS.items():
        if target_model not in (tag, "all"):
            continue

        optimized = load_optimized_params(tag, dataset_tag)
        if optimized:
            params = cfg["base_params"].copy()
            params.update(cfg.get("fixed_params", {}))
            params.update(optimized)
        else:
            params = cfg["base_params"].copy()

        model_params[tag] = params

    return model_params


def recalculate_cv_summary_with_optimal_threshold(trainer: Trainer, optimal_thresholds: dict, dataset_tag: str) -> pd.DataFrame:
    rows = []
    for result in trainer.fold_results:
        fold_id = result["fold"]
        row = {"fold": fold_id, "dataset_tag": dataset_tag}

        X_val_scaled = result["X_val"]
        y_val = result["y_val"]

        for model_name in MODEL_TAGS:
            if model_name not in result["models"]:
                continue

            model = result["models"][model_name]
            threshold = optimal_thresholds.get(model_name, 0.5)

            evaluator = Evaluator(threshold=threshold)
            metrics = evaluator.evaluate(model, X_val_scaled, y_val)

            row[f"{model_name}_prauc"] = metrics["prauc"]
            row[f"{model_name}_precision"] = metrics["precision"]
            row[f"{model_name}_recall"] = metrics["recall"]
            row[f"{model_name}_f1"] = metrics["f1"]
            row[f"{model_name}_threshold"] = metrics["threshold"]

        rows.append(row)
    return pd.DataFrame(rows)


def run_optimization(dataset_tag: str, target_model: str = "all", optimize_metric: str = "prauc") -> dict:
    use_clean = dataset_tag == "clean"
    logger.info(
        f"Iniciando optimización de hiperparámetros | Dataset: {dataset_tag.upper()} | "
        f"Modelo(s): {target_model.upper()} | Métrica objetivo: {optimize_metric.upper()}"
    )

    loader = DataLoader(use_clean=use_clean).load().split()
    if loader.X_train is None or loader.y_train is None:
        logger.error("Intento de optimización sin datos de entrenamiento.")
        raise RuntimeError("X_train / y_train no disponibles. Ejecuta split() en DataLoader.")

    cv_splits = loader.get_cv_folds()

    best_params = {}

    for tag, cfg in MODEL_CONFIGS.items():
        if target_model not in (tag, "all"):
            continue
        logger.info(f"Optimizando {cfg['display_name']}...")
        opt = HyperparameterOptimizer(
            study_name=f"{tag}_optimization", direction="maximize",
            n_trials=OPTUNA_TRIALS, dataset_tag=dataset_tag
        )
        best_params[tag] = opt.optimize_cv(
            tag, cfg["param_space"], loader.X_train, loader.y_train,
            cv_splits, Preprocessor, metric=optimize_metric,
            patience=OPTUNA_PATIENCE, tolerance=OPTUNA_TOLERANCE
        )  # type: ignore

    logger.info(f"Optimización completada para '{dataset_tag}'.")
    return best_params


def run_history_plot(model_tag: str, dataset_tag: str) -> None:
    models = list(MODEL_TAGS) if model_tag == "all" else [model_tag]
    datasets = list(DATASET_TAGS) if dataset_tag == "all" else [dataset_tag]

    for m in models:
        for d in datasets:
            csv_path = RESULTS_CLF_DIR / f"{m}_optimization_trials_history_{d}.csv"
            logger.info(f"Generando historial gráfico: {m.upper()} — Dataset: '{d}'")
            out_path = plot_optuna_history(csv_path=csv_path, model_tag=m, dataset_tag=d)
            if out_path:
                logger.info(f"Gráfica de Optuna guardada en: {out_path.name}")


def run_pipeline(dataset_tag: str, target_model: str, thresh_metric: str, skip_cv: bool) -> dict:
    use_clean = dataset_tag == "clean"
    logger.info(f"Iniciando Pipeline | Dataset: {dataset_tag.upper()} | Modelo objetivo: {target_model.upper()}")

    loader = DataLoader(use_clean=use_clean).load().split()

    model_params = build_model_params(target_model, dataset_tag)
    trainer = Trainer(loader=loader, dataset_tag=dataset_tag, model_params=model_params)

    if not skip_cv:
        logger.info(f"Ejecutando Validación Cruzada ({N_FOLDS} folds)...")
        trainer.run_cv()
        for result in trainer.fold_results:
            Persistence.save_fold_metrics(metrics=result["metrics"], fold_id=result["fold"], dataset_tag=dataset_tag)
            for m_name, m_obj in result["models"].items():
                Persistence.save_model(model=m_obj, model_name=m_name, fold_id=result["fold"], dataset_tag=dataset_tag)
        summary = trainer.get_cv_summary()
        Persistence.save_cv_summary(summary, dataset_tag)
    else:
        logger.warning("Validación cruzada omitida por argumento --skip-cv.")
        summary = None
        trainer.fold_results = []

    logger.info("Entrenando modelos finales sobre el total de X_train (70%)...")
    trainer.train_final()

    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        Persistence.save_model(model=model, model_name=model_name, fold_id="final", dataset_tag=dataset_tag)

    logger.info("Evaluando métricas base en X_val (20%) con umbral por defecto (0.5)...")
    prep_final = trainer.final_models["preprocessor"]
    X_val_scaled = prep_final.transform(loader.X_val)
    X_tune_scaled = prep_final.transform(loader.X_tune)

    evaluator = Evaluator()
    val_metrics = {
        model_name: evaluator.evaluate(model, X_val_scaled, loader.y_val)  # type: ignore
        for model_name, model in trainer.final_models.items()
        if model_name not in ("preprocessor", "weights")
    }

    logger.info(f"Optimizando umbrales en X_tune (10%) basado en estrategia: {thresh_metric.upper()}...")
    optimal_thresholds = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator()
        threshold = opt_eval.optimize_threshold(model=model, X=X_tune_scaled, y=loader.y_tune, strategy=thresh_metric)  # type: ignore
        optimal_thresholds[model_name] = threshold
        logger.info(f"{model_name.upper()} → Umbral óptimo: {threshold:.3f}")

    logger.info("Evaluando métricas finales en X_tune (10%) con umbral optimizado...")
    tune_metrics = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        tune_metrics[model_name] = opt_eval.evaluate(model, X_tune_scaled, loader.y_tune)  # type: ignore

    logger.info("Evaluando métricas finales en X_val (20%) con umbral optimizado...")
    val_metrics_opt = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"):
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        val_metrics_opt[model_name] = opt_eval.evaluate(model, X_val_scaled, loader.y_val)  # type: ignore

    for model_name, metrics in val_metrics_opt.items():
        logger.info(f"[{model_name.upper()}] PR-AUC: {metrics['prauc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    logger.info("Persistiendo métricas y artefactos finales...")
    final_results = {
        "dataset_tag": dataset_tag,
        "thresh_metric": thresh_metric,
        "optimal_thresholds": optimal_thresholds,
        "val_metrics": val_metrics,
        "val_metrics_opt": val_metrics_opt,
        "tune_metrics": tune_metrics,
        "class_weights": {str(k): v for k, v in trainer.final_models["weights"].items()},
    }
    Persistence.save_final_results(final_results, dataset_tag, thresh_metric)

    summary_optimal = None
    if not skip_cv:
        logger.info("Recalculando el sumario CV aplicando umbrales óptimos...")
        summary_optimal = recalculate_cv_summary_with_optimal_threshold(trainer, optimal_thresholds, dataset_tag)

    return {
        "summary": summary,
        "summary_optimal": summary_optimal,
        "final_results": final_results,
        "trainer": trainer,
        "loader": loader,
        "prep_final": prep_final,
        "thresh_metric": thresh_metric,
    }


def load_pipeline_results(dataset_tag: str, thresh_metric: str) -> dict:
    logger.info(f"Artefactos existentes detectados para '{dataset_tag}' (estrategia='{thresh_metric}'). Evitando re-entrenamiento...")
    return {
        "summary": Persistence.load_cv_summary(dataset_tag),
        "final_results": Persistence.load_final_results(dataset_tag, thresh_metric),
        "trainer": None, "loader": None, "prep_final": None,
        "thresh_metric": thresh_metric,
    }


def run_visualizations(results: dict, dataset_tag: str, skip_cv: bool) -> None:
    summary = results.get("summary")
    summary_optimal = results.get("summary_optimal")
    trainer = results.get("trainer")
    loader = results.get("loader")
    prep = results.get("prep_final")
    final_results = results.get("final_results")
    thresh_metric = results.get("thresh_metric")

    if not skip_cv and summary is not None:
        plot_fold_metrics(summary, dataset_tag)
        if summary_optimal is not None:
            plot_fold_metrics(summary_optimal, dataset_tag, is_optimal=True, thresh_metric=thresh_metric)
        if trainer:
            plot_class_weights(trainer.fold_results, dataset_tag)

    if loader and trainer and prep:
        X_val_scaled = prep.transform(loader.X_val)
        curve_data = {
            m_name: Evaluator().evaluate_curve(model, X_val_scaled, loader.y_val)
            for m_name, model in trainer.final_models.items() if m_name not in ("preprocessor", "weights")
        }
        plot_pr_curves(curve_data, dataset_tag)

    if final_results:
        path = plot_final_metrics(final_results, dataset_tag)
        if path:
            logger.info(f"Gráfica de métricas finales Test guardada: {path.name}")


def run_confusion_matrices(dataset_tag: str, target_model: str, partition: str, thresh_metric: str, threshold: float | None) -> None:
    """
    Genera la(s) matriz(ces) de confusión para el/los modelo(s) indicados
    usando ModelInference, que recupera modelo + scaler desde disco y
    resuelve el umbral (manual si se indica, óptimo persistido si no).
    Funciona igual con artefactos recién entrenados o cargados de caché,
    a diferencia de la versión anterior que requería trainer/loader en memoria.
    """
    inference = ModelInference()
    models = list(MODEL_TAGS) if target_model == "all" else [target_model]

    for model_tag in models:
        try:
            inference.confusion_matrix(
                dataset_tag=dataset_tag, model_tag=model_tag, partition=partition,
                threshold=threshold, thresh_metric=thresh_metric,
            )
        except FileNotFoundError as e:
            logger.warning(f"No se pudo generar matriz de confusión [{dataset_tag}/{model_tag}]: {e}")

if __name__ == "__main__":
    args = parse_args()
    tags = list(DATASET_TAGS) if args.dataset == "all" else [args.dataset]
    all_data = {}

    logger.info("=== Ejecución Iniciada: Clasificación Binaria ===")

    if args.optimize:
        expected = ["optimize", "model", "dataset", "optimize_metric"]
        validate_context_args(args, expected, "OPTIMIZE")
        
        for tag in tags:
            run_optimization(tag, target_model=args.model, optimize_metric=args.optimize_metric)
            
        logger.info("Operación completada: Optimización de hiperparámetros finalizada.")
        sys.exit(0)

    if args.plot_history:
        expected = ["plot_history", "model", "dataset"]
        validate_context_args(args, expected, "PLOT HISTORY")
        
        for tag in tags:
            run_history_plot(args.model, tag) 
            
        logger.info("Operación completada: Gráficas de historial generadas.")
        sys.exit(0)
        

    if args.plot_cm:
        expected = ["plot_cm", "model", "dataset", "partition", "thresh_metric", "threshold"]
        validate_context_args(args, expected, "CONFUSION MATRIX")
        
        logger.info(
            f"Generando matriz(ces) | Partición: {args.partition} | "
            f"Modelo: {args.model} | Estrategia: {args.thresh_metric} | "
            f"Umbral manual: {args.threshold if args.threshold is not None else 'N/A'}"
        )
        for tag in tags:
            run_confusion_matrices(tag, args.model, args.partition, args.thresh_metric, args.threshold)
            
        logger.info("Operación completada: Matrices de confusión generadas.")
        sys.exit(0)

    # ---------------------------------------------------------
    # FLUJO PRINCIPAL: Entrenamiento, Evaluación y Comparación
    # ---------------------------------------------------------
    expected = ["dataset", "model", "thresh_metric", "skip_cv", "force_retrain"]
    validate_context_args(args, expected, "MAIN PIPELINE")

    for tag in tags:
        if not args.force_retrain and not args.skip_cv and artifacts_exist(tag, args.thresh_metric):
            all_data[tag] = load_pipeline_results(tag, args.thresh_metric)
        else:
            all_data[tag] = run_pipeline(tag, args.model, args.thresh_metric, args.skip_cv)

    logger.info("Generando etapa de visualizaciones estáticas...")
    for tag in tags:
        run_visualizations(all_data[tag], tag, args.skip_cv)

    if len(all_data) == 2:
        if not args.skip_cv and all(all_data[t].get("summary") is not None for t in DATASET_TAGS):
            plot_dataset_comparison(all_data["full"]["summary"], all_data["clean"]["summary"])
            if all(all_data[t].get("summary_optimal") is not None for t in DATASET_TAGS):
                plot_dataset_comparison(
                    all_data["full"]["summary_optimal"], all_data["clean"]["summary_optimal"],
                    is_optimal=True, thresh_metric=args.thresh_metric,
                )

        if all(all_data[t].get("final_results") is not None for t in DATASET_TAGS):
            path = plot_final_comparison(all_data["full"]["final_results"], all_data["clean"]["final_results"])
            if path:
                logger.info(f"Gráfica comparativa Test (Full vs Clean) guardada: {path.name}")

    logger.info("=== Pipeline Completado con Éxito ===")
    sys.exit(0)