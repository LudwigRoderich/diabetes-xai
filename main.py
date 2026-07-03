import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from src.grouper import ProbabilityGrouper
from src.config import CONTINUOUS_COLS, ORDINAL_COLS, TARGET_COL

from src.config import (
    N_FOLDS, RESULTS_DIR, OPTUNA_TRIALS, OPTUNA_PATIENCE, OPTUNA_TOLERANCE,
    dt_param_space, xgb_param_space, DT_BASE_PARAMS, XGBOOST_BASE_PARAMS,
    get_logger
)
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.persistence import Persistence
from src.trainer import Trainer
from src.optimizer import HyperparameterOptimizer
from src.preprocessor import Preprocessor
from src.visualizer import (
    plot_class_weights, plot_dataset_comparison, plot_fold_metrics,
    plot_pr_curves, plot_final_metrics, plot_final_comparison, plot_optuna_history, plot_proba_histogram
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline DT y XGBoost para predicción de diabetes.")
    parser.add_argument("--force-retrain", action="store_true", default=False)
    parser.add_argument("--dataset", choices=["full", "clean", "both"], default="both")
    parser.add_argument("--optimize", action="store_true", default=False)
    parser.add_argument("--optimize-model", choices=["dt", "xgb", "both"], default="both")
    parser.add_argument("--model", choices=["dt", "xgb", "both"], default="both")
    parser.add_argument("--thresh-metric", choices=["prauc", "f1", "recall"], default="prauc")
    parser.add_argument("--skip-cv", action="store_true", default=False)
    parser.add_argument("--plot-history", action="store_true", default=False)
    parser.add_argument("--minimize", action="store_true", default=False)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--plot-proba", action="store_true", default=False)
    parser.add_argument("--group-analysis", action="store_true", default=False)
    parser.add_argument("--group-min", type=float, default=0.0)
    parser.add_argument("--group-max", type=float, default=0.25)
    parser.add_argument("--anchor-prob", type=float, default=0.125)
    parser.add_argument("--radius", type=float, default=0.3)
    
    return parser.parse_args()

def load_optimized_params(model_name: str, dataset_tag: str) -> dict | None:
    filepath = RESULTS_DIR / f"{model_name}_optimization_best_params_{dataset_tag}.json"
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Parámetros óptimos cargados para {model_name.upper()} ({dataset_tag})")
                return data.get("best_params", None)
        except Exception as e:
            logger.error(f"Error al leer parámetros en {filepath.name}: {e}")
    return None

def artifacts_exist(dataset_tag: str) -> bool:
    cv_summary = RESULTS_DIR / f"cv_summary_{dataset_tag}.csv"
    final_res = RESULTS_DIR / f"final_results_{dataset_tag}.json"
    return cv_summary.exists() and final_res.exists()

def recalculate_cv_summary_with_optimal_threshold(trainer: Trainer, optimal_thresholds: dict, dataset_tag: str) -> pd.DataFrame:
    rows = []
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
            
            evaluator = Evaluator(threshold=threshold)
            metrics = evaluator.evaluate(model, X_val_scaled, y_val)
            
            row[f"{model_name}_prauc"] = metrics["prauc"]
            row[f"{model_name}_precision"] = metrics["precision"]
            row[f"{model_name}_recall"] = metrics["recall"]
            row[f"{model_name}_f1"] = metrics["f1"]
            row[f"{model_name}_threshold"] = metrics["threshold"]
            
        rows.append(row)
    return pd.DataFrame(rows)

def run_optimization(dataset_tag: str, optimize_model: str = "both") -> dict:
    use_clean = dataset_tag == "clean"
    logger.info(f"Iniciando optimización de hiperparámetros | Dataset: {dataset_tag.upper()}")

    loader = DataLoader(use_clean=use_clean).load().split()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=141421)
    cv_splits = list(cv.split(loader.X_train, loader.y_train)) #type: ignore
    
    best_params = {}
    
    if optimize_model in ("dt", "both"):
        logger.info("Optimizando Decision Tree...")
        opt_dt = HyperparameterOptimizer(study_name="dt_optimization", direction="maximize", n_trials=OPTUNA_TRIALS, dataset_tag=dataset_tag)
        best_params["dt"] = opt_dt.optimize_cv(DecisionTreeClassifier, dt_param_space, loader.X_train, loader.y_train, cv_splits, Preprocessor, metric="prauc", patience=OPTUNA_PATIENCE, tolerance=OPTUNA_TOLERANCE) #type: ignore
    
    if optimize_model in ("xgb", "both"):
        logger.info("Optimizando XGBoost...")
        opt_xgb = HyperparameterOptimizer(study_name="xgb_optimization", direction="maximize", n_trials=OPTUNA_TRIALS, dataset_tag=dataset_tag)
        best_params["xgb"] = opt_xgb.optimize_cv(XGBClassifier, xgb_param_space, loader.X_train, loader.y_train, cv_splits, Preprocessor, metric="prauc", patience=OPTUNA_PATIENCE, tolerance=OPTUNA_TOLERANCE) #type: ignore
    
    logger.info(f"Optimización completada para '{dataset_tag}'.")
    return best_params

def run_history_plot(model_tag: str, dataset_tag: str) -> None:
    models = ["dt", "xgb"] if model_tag == "both" else [model_tag]
    datasets = ["full", "clean"] if dataset_tag == "both" else [dataset_tag]

    for m in models:
        for d in datasets:
            csv_path = RESULTS_DIR / f"{m}_optimization_trials_history_{d}.csv"
            logger.info(f"Generando historial gráfico: {m.upper()} — Dataset: '{d}'")
            out_path = plot_optuna_history(csv_path=csv_path, model_tag=m, dataset_tag=d)
            if out_path:
                logger.info(f"Gráfica de Optuna guardada en: {out_path.name}")

def run_group_analysis(args: argparse.Namespace) -> None:
    
    tag = "clean" if args.dataset in ["clean", "both"] else "full"
    use_model = args.model if args.model else "xgb"
    loader = DataLoader(use_clean=(tag == "clean")).load().split()
    if loader.X_val is None:
        logger.error(f"Dataset de validacion no cargado para '{tag}'. Asegúrate de ejecutar el pipeline de clasificación primero.")
        sys.exit(1)
    
    #model = Persistence.load_model("xgb", fold_id=0, dataset_tag=tag) 
    model = Persistence.load_model(use_model, fold_id=0, dataset_tag=tag)
    evaluator = Evaluator()
    probas = evaluator._get_proba(model, loader.X_val)
    
    
    grouper = ProbabilityGrouper(loader.X_val, probas)
    #grouper = ProbabilityGrouper(data, probas)
    
    subgroup = grouper.get_subgroup(args.group_min, args.group_max)
    anchor = grouper.get_anchor(subgroup, args.anchor_prob)
    feature_cols = CONTINUOUS_COLS + ORDINAL_COLS
    
    neighborhood = grouper.get_neighborhood(subgroup, anchor, args.radius, feature_cols)
    neighborhood.to_csv(RESULTS_DIR / f"neighborhood_center_{args.anchor_prob}_radius_{args.radius or 0.0}_{tag}.csv", index=False)
    logger.info("Análisis de agrupamiento finalizado y exportado.")

def run_pipeline(dataset_tag: str, target_model: str, thresh_metric: str, skip_cv: bool) -> dict:
    use_clean = dataset_tag == "clean"
    logger.info(f"Iniciando Pipeline | Dataset: {dataset_tag.upper()} | Modelo objetivo: {target_model.upper()}")

    loader = DataLoader(use_clean=use_clean).load().split()

    dt_opt = load_optimized_params("dt", dataset_tag) if target_model in ["dt", "both"] else None
    xgb_opt = load_optimized_params("xgb", dataset_tag) if target_model in ["xgb", "both"] else None

    if dt_opt:
        dt_params = DT_BASE_PARAMS.copy()
        dt_params.update(dt_opt)
    else:
        dt_params = None

    if xgb_opt:
        xgb_params = XGBOOST_BASE_PARAMS.copy()
        xgb_params.update({
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "aucpr"
        })
        xgb_params.update(xgb_opt)
    else:
        xgb_params = None

    trainer = Trainer(loader=loader, dataset_tag=dataset_tag, dt_params=dt_params, xgb_params=xgb_params)

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
    
    models_to_keep = ["preprocessor", "weights"]
    if target_model in ["dt", "both"]: 
        models_to_keep.append("dt")
    if target_model in ["xgb", "both"]: 
        models_to_keep.append("xgb")
        
    trainer.final_models = {k: v for k, v in trainer.final_models.items() if k in models_to_keep}

    for model_name, model in trainer.final_models.items():
        if model_name == "preprocessor": 
            continue
        Persistence.save_model(model=model, model_name=model_name, fold_id="final", dataset_tag=dataset_tag)

    logger.info("Evaluando métricas base en X_val (20%) con umbral por defecto (0.5)...")
    prep_final = trainer.final_models["preprocessor"]
    X_val_scaled = prep_final.transform(loader.X_val)
    X_tune_scaled = prep_final.transform(loader.X_tune)

    evaluator = Evaluator()
    val_metrics = {
        model_name: evaluator.evaluate(model, X_val_scaled, loader.y_val) #type: ignore
        for model_name, model in trainer.final_models.items()
        if model_name not in ("preprocessor", "weights")
    }

    logger.info(f"Optimizando umbrales en X_tune (10%) basado en métrica: {thresh_metric.upper()}...")
    optimal_thresholds = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"): 
            continue
        opt_eval = Evaluator()
        threshold = opt_eval.optimize_threshold(model=model, X=X_tune_scaled, y=loader.y_tune, strategy=thresh_metric) #type: ignore
        optimal_thresholds[model_name] = threshold
        logger.info(f"{model_name.upper()} → Umbral óptimo: {threshold:.3f}")

    logger.info("Evaluando métricas finales en X_tune (10%) con umbral optimizado...")
    tune_metrics = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"): 
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        tune_metrics[model_name] = opt_eval.evaluate(model, X_tune_scaled, loader.y_tune) #type: ignore

    logger.info("Evaluando métricas finales en X_val (20%) con umbral optimizado...")
    val_metrics_opt = {}
    for model_name, model in trainer.final_models.items():
        if model_name in ("preprocessor", "weights"): 
            continue
        opt_eval = Evaluator(threshold=optimal_thresholds[model_name])
        val_metrics_opt[model_name] = opt_eval.evaluate(model, X_val_scaled, loader.y_val) #type: ignore
        
    for model_name, metrics in val_metrics_opt.items():
        logger.info(f"[{model_name.upper()}] PR-AUC: {metrics['prauc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    logger.info("Persistiendo métricas y artefactos finales...")
    final_results = {
        "dataset_tag": dataset_tag,
        "optimal_thresholds": optimal_thresholds,
        "val_metrics": val_metrics,
        "val_metrics_opt": val_metrics_opt,
        "tune_metrics": tune_metrics,
        "class_weights": {str(k): v for k, v in trainer.final_models["weights"].items()},
    }
    Persistence.save_final_results(final_results, dataset_tag)

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
    }

def load_pipeline_results(dataset_tag: str) -> dict:
    logger.info(f"Artefactos existentes detectados para '{dataset_tag}'. Evitando re-entrenamiento...")
    return {
        "summary": Persistence.load_cv_summary(dataset_tag),
        "final_results": Persistence.load_final_results(dataset_tag),
        "trainer": None, "loader": None, "prep_final": None,
    }

def run_visualizations(results: dict, dataset_tag: str, skip_cv: bool) -> None:
    summary = results.get("summary")
    summary_optimal = results.get("summary_optimal")
    trainer = results.get("trainer")
    loader = results.get("loader")
    prep = results.get("prep_final")
    final_results = results.get("final_results")

    if not skip_cv and summary is not None:
        plot_fold_metrics(summary, dataset_tag)
        if summary_optimal is not None:
            plot_fold_metrics(summary_optimal, dataset_tag, is_optimal=True)
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

def execute_plot_proba(args: argparse.Namespace, tag: str) -> None:
    
    logger.info(f"Verificando estado de artefactos para el dataset: {tag}")
    
    if not artifacts_exist(tag) or args.force_retrain:
        logger.warning(f"Artefactos faltantes para '{tag}'. Ejecutando entrenamiento previo...")
        run_pipeline(tag, args.model, args.thresh_metric, args.skip_cv)
        
    try:
        
        loader = DataLoader(use_clean=(tag == "clean")).load().split()
        if loader.df is None:
            logger.error(f"Dataset no cargado para '{tag}'. Asegúrate de ejecutar el pipeline de clasificación primero.")
            sys.exit(1)
        use_model = args.model if args.model else "xgb"
        model = Persistence.load_model(use_model, fold_id=0, dataset_tag=tag)
        
        try:
            #data = loader.df.copy().drop(columns=[TARGET_COL])

            preprocessor = Preprocessor.load(fold_id=0, dataset_tag=tag)
            if loader.X_val is None:
            # if loader.df is None:
                logger.error(f"Datos de validación no disponibles para '{tag}'. Asegúrate de ejecutar el pipeline de clasificación primero.")
                sys.exit(1)
            X_data = loader.X_val.copy()
            #X_data = loader.df.copy()
            X_data[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(loader.X_val[CONTINUOUS_COLS])
            #data[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(data[CONTINUOUS_COLS])
        except FileNotFoundError:
            X_data = loader.X_val
            #logger.warning(f"Preprocesador no encontrado para '{tag}'. Usando datos sin escalar para el histograma de probabilidades.")

        evaluator = Evaluator()
        probas = evaluator._get_proba(model, X_data)
        
        plot_proba_histogram(probas, use_model, tag)
        logger.info(f"Histograma de probabilidades para '{tag}' generado correctamente.")
        
    except Exception as e:
        logger.error(f"Fallo durante la ejecución de plot_proba para {tag}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    tags = ["full", "clean"] if args.dataset == "both" else [args.dataset]
    all_data = {}

    logger.info("=== Ejecución Iniciada: Diabetes XAI Pipeline ===")

    if args.plot_proba:
        for tag in tags:
            execute_plot_proba(args, tag)
        sys.exit(0)

    if args.group_analysis:
        run_group_analysis(args)
        sys.exit(0)

    if args.plot_history:
        run_history_plot(args.model, args.dataset)
        logger.info("Operación completada: Gráficas de historial generadas.")
        sys.exit(0)

    if args.optimize:
        for tag in tags:
            run_optimization(tag, optimize_model=args.optimize_model)
        logger.info("Operación completada: Optimización de hiperparámetros finalizada.")
    else:
        for tag in tags:
            if not args.force_retrain and not args.skip_cv and artifacts_exist(tag):
                all_data[tag] = load_pipeline_results(tag)
            else:
                all_data[tag] = run_pipeline(tag, args.model, args.thresh_metric, args.skip_cv)

        logger.info("Generando etapa de visualizaciones estáticas...")
        for tag in tags:
            run_visualizations(all_data[tag], tag, args.skip_cv)

        if len(all_data) == 2:
            if not args.skip_cv and all(all_data[t].get("summary") is not None for t in ["full", "clean"]):
                plot_dataset_comparison(all_data["full"]["summary"], all_data["clean"]["summary"])
                if all(all_data[t].get("summary_optimal") is not None for t in ["full", "clean"]):
                    plot_dataset_comparison(all_data["full"]["summary_optimal"], all_data["clean"]["summary_optimal"], is_optimal=True)
            
            if all(all_data[t].get("final_results") is not None for t in ["full", "clean"]):
                path = plot_final_comparison(all_data["full"]["final_results"], all_data["clean"]["final_results"])
                if path:
                    logger.info(f"Gráfica comparativa Test (Full vs Clean) guardada: {path.name}")

        logger.info("=== Pipeline Completado con Éxito ===")

    