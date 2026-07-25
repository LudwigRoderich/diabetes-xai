"""
Punto de entrada de la etapa XAI del proyecto MM-700.

Carga los artefactos del pipeline de clasificación (modelos y preprocesador
del fold indicado) y orquesta los análisis de explicabilidad configurados
por CLI.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.config import (
    XAI_DIR,
    CONTINUOUS_COLS,
    MODEL_TAGS,
    DATASET_TAGS,
    LIME_SURROGATE_CHOICES,
    XAI_METHOD_CHOICES,
    get_logger,
)
from src.binary_clasification.data_loader import DataLoader
from src.binary_clasification.persistence import Persistence
from src.binary_clasification.preprocessor import Preprocessor
from src.xai.local_helper import LocalStabilityAnalyzer
from src.xai.pfi import PermutationImportanceAnalyzer
from src.xai.treeShap import TreeShapLocalAnalyzer
from src.xai.lime import LimeLocalAnalyzer

logger = get_logger("xai_main")

XAI_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa XAI — análisis de explicabilidad sobre modelos finales de clasificación binaria."
    )
    parser.add_argument("--dataset", choices=list(DATASET_TAGS) + ["both"], default="both")
    parser.add_argument("--model", choices=list(MODEL_TAGS) + ["all"], default="xgb")
    parser.add_argument(
        "--method",
        choices=list(XAI_METHOD_CHOICES) + ["all"],
        default="intrinsic",
        help="Método de explicabilidad a ejecutar. 'intrinsic' extrae la importancia base del modelo.",
    )
    parser.add_argument(
        "--fold-id",
        type=str,
        default="final",
        help="ID del fold cuyos artefactos se cargan. 'final' (defecto) para el modelo re-entrenado sobre todo X_train, o un número (0-N) para un fold específico de CV.",
    )
    parser.add_argument(
        "--group-file",
        type=str,
        default=None,
        help="Ruta al archivo CSV del vecindario generado por grouper_main.py (requerido para métodos locales como shap y lime).",
    )
    parser.add_argument(
        "--lime-surrogate",
        choices=list(LIME_SURROGATE_CHOICES),
        default=LIME_SURROGATE_CHOICES[0],
        help="Modelo surrogate interpretable usado por LIME: 'ridge' (lineal, con signo) o 'tree' (no lineal, sin signo).",
    )
    parser.add_argument(
        "--plot-stability",
        action="store_true",
        default=False,
        help="Graficar el panel de dispersión de un archivo de estabilidad previamente generado.",
    )
    parser.add_argument(
        "--plot-distribution",
        action="store_true",
        default=False,
        help="Graficar histogramas y boxplots sobre la distribución de la inestabilidad.",
    )
    parser.add_argument(
        "--summarize-stability",
        action="store_true",
        default=False,
        help="Generar un reporte CSV con estadísticas descriptivas (media, varianza, percentiles) del archivo de estabilidad.",
    )
    parser.add_argument(
        "--stability-file",
        type=str,
        default=None,
        help="Ruta al archivo CSV de estabilidad (ej. shap_stability_xgb_clean.csv).",
    )
    return parser.parse_args()


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


def load_xai_artifacts(dataset_tag: str, fold_id: str) -> dict:
    try:
        models = {
            model_name: Persistence.load_model(model_name, fold_id=fold_id, dataset_tag=dataset_tag)
            for model_name in MODEL_TAGS
        }
        preprocessor = Preprocessor.load(fold_id=fold_id, dataset_tag=dataset_tag)

        loader = DataLoader(use_clean=(dataset_tag == "clean")).load().split()
        if loader.X_val is None or loader.y_val is None or loader.X_val.empty or loader.y_val.empty:
            logger.error(f"Conjunto de validación vacío para '{dataset_tag}'. Verificar la etapa de preprocesamiento.")
            sys.exit(1)

        X_val_scaled = loader.X_val.copy()
        X_val_scaled[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(loader.X_val[CONTINUOUS_COLS])

        return {
            "dataset_tag": dataset_tag,
            "fold_id": fold_id,
            "models": models,
            "preprocessor": preprocessor,
            "X_val_raw": loader.X_val,
            "X_val_scaled": X_val_scaled,
            "y_val": loader.y_val,
        }
    except FileNotFoundError as e:
        logger.error(f"Artefactos faltantes para {dataset_tag}. Error: {e}")
        sys.exit(1)


def extract_intrinsic_importance(artifacts: dict, model_name: str) -> pd.DataFrame | None:
    dataset_tag = artifacts["dataset_tag"]
    model = artifacts["models"].get(model_name)
    X_scaled = artifacts["X_val_scaled"]

    if model is None or not hasattr(model, "feature_importances_"):
        return None

    # Bloque espaciador
    logger.info("\n\n" + "=" * 80)
    logger.info("=== NUEVA EXTRACCIÓN INTRÍNSECA ===")
    logger.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modelo: {model_name.upper()} | Dataset: {dataset_tag}")
    logger.info("=" * 80 + "\n")

    importances = model.feature_importances_
    features = X_scaled.columns.tolist()

    df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    df["rank"] = df.index + 1

    fold_id = artifacts.get("fold_id", 0)
    out_path = XAI_DIR / f"intrinsic_importance_{model_name}_{dataset_tag}_fold{fold_id}.csv"
    df.to_csv(out_path, index=False)
    
    logger.info("\n" + "-" * 50)
    logger.info("RESULTADOS INTRÍNSECOS:")
    logger.info(f"Referencia interpretativa guardada en: {out_path.name}")
    logger.info("-" * 50 + "\n\n")

    return df


def run_pfi(artifacts: dict, model_name: str) -> pd.DataFrame | None:
    dataset_tag = artifacts["dataset_tag"]
    model = artifacts["models"].get(model_name)

    if model is None:
        return None

    analyzer = PermutationImportanceAnalyzer(
        model=model,
        model_name=model_name,
        dataset_tag=dataset_tag,
    )

    analyzer.compute(X=artifacts["X_val_scaled"], y=artifacts["y_val"])
    importance_df = analyzer.get_importance_df()

    fold_id = artifacts.get("fold_id", 0)
    out_path = XAI_DIR / f"pfi_results_{model_name}_{dataset_tag}_fold{fold_id}_{analyzer.n_repeats}reps_seed{analyzer.random_state}.csv"
    importance_df.to_csv(out_path, index=False)
    logger.info(f"[PFI] Resultados CSV exportados a: {out_path.name}")

    return importance_df


def run_shap_local(artifacts: dict, model_name: str, group_file: Path) -> pd.DataFrame:
    model = artifacts["models"].get(model_name)
    if model is None:
        return pd.DataFrame()

    preprocessor = artifacts["preprocessor"]
    feature_cols = artifacts["X_val_raw"].columns.tolist()

    analyzer = TreeShapLocalAnalyzer(
        model=model,
        model_name=model_name,
        dataset_tag=artifacts["dataset_tag"],
    )

    return analyzer.analyze_stability(
        neighborhood_path=group_file,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
    )


def run_lime_local(artifacts: dict, model_name: str, group_file: Path, surrogate: str) -> pd.DataFrame:
    model = artifacts["models"].get(model_name)
    if model is None:
        return pd.DataFrame()

    preprocessor = artifacts["preprocessor"]
    feature_cols = artifacts["X_val_raw"].columns.tolist()

    analyzer = LimeLocalAnalyzer(
        model=model,
        model_name=model_name,
        dataset_tag=artifacts["dataset_tag"],
        surrogate=surrogate,
    )

    return analyzer.analyze_stability(
        neighborhood_path=group_file,
        feature_cols=feature_cols,
        reference_df=artifacts["X_val_raw"],
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.plot_stability or args.plot_distribution or args.summarize_stability:
        expected = ["plot_stability", "plot_distribution", "summarize_stability", "stability_file"]
        validate_context_args(args, expected, "STABILITY ANALYSIS")
        
        if not args.stability_file:
            logger.error("Debe proporcionar la ruta del CSV con --stability-file para ejecutar el análisis post-ejecución.")
            sys.exit(1)

        file_path = Path(args.stability_file)
        logger.info(f"Iniciando herramientas de análisis para: {file_path.name}")

        if args.summarize_stability:
            LocalStabilityAnalyzer.compute_stability_summary(file_path)
            
        if args.plot_stability:
            LocalStabilityAnalyzer.plot_stability_scatter(file_path)
            
        if args.plot_distribution:
            LocalStabilityAnalyzer.plot_stability_distribution(file_path)
            
        logger.info("Operaciones de reporte/gráficos finalizadas.")
        sys.exit(0)

    expected = ["dataset", "model", "method", "fold_id", "lime_surrogate", "group_file"]
    validate_context_args(args, expected, "MAIN XAI PIPELINE")

    tags = list(DATASET_TAGS) if args.dataset == "both" else [args.dataset]
    model_names = list(MODEL_TAGS) if args.model == "all" else [args.model]

    logger.info("=== Etapa XAI Iniciada ===")
    logger.info(
        f"Parámetros de ejecución: dataset={args.dataset}, model={args.model}, method={args.method}, "
        f"fold_id={args.fold_id}, lime_surrogate={args.lime_surrogate}, group_file={args.group_file}"
    )

    for tag in tags:
        artifacts = load_xai_artifacts(tag, args.fold_id)

        if args.method == "intrinsic":
            for name in model_names:
                extract_intrinsic_importance(artifacts, name)

        if args.method in ("pfi", "all"):
            for name in model_names:
                run_pfi(artifacts, name)

        if args.method in ("shap", "all"):
            if args.group_file:
                for name in model_names:
                    run_shap_local(artifacts, name, Path(args.group_file))

        if args.method in ("lime", "all"):
            if args.group_file:
                for name in model_names:
                    run_lime_local(artifacts, name, Path(args.group_file), args.lime_surrogate)

    logger.info("=== Etapa XAI Completada ===")