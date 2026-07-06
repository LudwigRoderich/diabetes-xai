"""
Punto de entrada de la etapa XAI del proyecto MM-700.

Carga los artefactos del pipeline de clasificación (modelos y preprocesador del
fold 'final') y orquesta los análisis de explicabilidad configurados por CLI.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from src.config import ORDINAL_COLS, XAI_DIR, get_logger, CONTINUOUS_COLS, RESULTS_XAI_DIR
from src.data_loader import DataLoader
from src.persistence import Persistence
from src.preprocessor import Preprocessor
from src.xai.local_helper import LocalStabilityAnalyzer
from src.xai.pfi import PermutationImportanceAnalyzer
from src.xai.shapley import ShapLocalAnalyzer

logger = get_logger(__name__)

XAI_DIR.mkdir(parents=True, exist_ok=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa XAI — análisis de explicabilidad sobre modelos finales del pipeline MM-700."
    )
    parser.add_argument("--dataset", choices=["full", "clean", "both"], default="both")
    parser.add_argument("--model", choices=["dt", "xgb", "both"], default="xgb")
    parser.add_argument(
        "--method",
        choices=["intrinsic", "pfi", "surrogate", "lime", "shap", "all"],
        default="all",
        help="Método de explicabilidad a ejecutar. 'intrinsic' extrae la importancia base del modelo."
    )
    parser.add_argument(
        "--group-file", 
        type=str, 
        default=None, 
        help="Ruta al archivo CSV del vecindario generado por grouper_main.py (Requerido para métodos locales)."
    )
    parser.add_argument(
        "--plot-stability", 
        action="store_true", 
        default=False, 
        help="Graficar el panel de dispersión de un archivo de estabilidad previamente generado."
    )
    parser.add_argument(
        "--stability-file", 
        type=str, 
        default=None, 
        help="Ruta al archivo CSV de estabilidad (ej. shap_stability_xgb_clean.csv)."
    )
    return parser.parse_args()

def load_xai_artifacts(dataset_tag: str) -> dict:
    """Carga de los modelos, el preprocesador y los datos del fold 0."""
    try:
        dt_model = Persistence.load_model("dt", fold_id=0, dataset_tag=dataset_tag)
        xgb_model = Persistence.load_model("xgb", fold_id=0, dataset_tag=dataset_tag)
        preprocessor = Preprocessor.load(fold_id=0, dataset_tag=dataset_tag)
        
        loader = DataLoader(use_clean=(dataset_tag == "clean")).load().split()
        if loader.X_val is None or loader.y_val is None or loader.X_val.empty or loader.y_val.empty:
            logger.error(f"Conjunto de validación vacío para '{dataset_tag}'. Verificar la etapa de preprocesamiento.")
            sys.exit(1)
        X_val_scaled = loader.X_val.copy()
        X_val_scaled[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(loader.X_val[CONTINUOUS_COLS])
        
        return {
            "dt_model": dt_model,
            "xgb_model": xgb_model,
            "preprocessor": preprocessor,
            "X_val_raw": loader.X_val,
            "X_val_scaled": X_val_scaled,
            "y_val": loader.y_val
        }
    except FileNotFoundError as e:
        logger.error(f"Artefactos faltantes para {dataset_tag}. Error: {e}")
        sys.exit(1)


def extract_intrinsic_importance(artifacts: dict, model_name: str) -> pd.DataFrame | None:
    """
    Extrae la importancia nativa (Gini/Impurity) del modelo si está soportada.
    Genera la línea base interpretativa para contrastar métodos post-hoc.
    """
    dataset_tag = artifacts["dataset_tag"]
    model = artifacts["models"].get(model_name)
    X_scaled = artifacts["X_val_scaled"]

    if model is None:
        logger.warning(f"[Intrinsic] Modelo {model_name.upper()} no disponible para '{dataset_tag}'.")
        return None

    if not hasattr(model, "feature_importances_"):
        logger.warning(f"[Intrinsic] El modelo {model_name.upper()} no expone 'feature_importances_'.")
        return None

    logger.info(f"[Intrinsic] Extrayendo importancia base de {model_name.upper()} ({dataset_tag})...")
    
    importances = model.feature_importances_
    features = X_scaled.columns.tolist()

    df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    df["rank"] = df.index + 1

    out_path = XAI_DIR / f"intrinsic_importance_{model_name}_{dataset_tag}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[Intrinsic] Referencia interpretativa guardada en: {out_path.name}")

    return df

def run_pfi(artifacts: dict, model_name: str) -> pd.DataFrame | None:
    """Ejecuta PFI para el modelo indicado y persiste los resultados."""
    dataset_tag = artifacts["dataset_tag"]
    model = artifacts["models"].get(model_name)

    if model is None:
        logger.warning(f"[PFI] Modelo {model_name.upper()} no disponible para '{dataset_tag}'. Saltando.")
        return None

    analyzer = PermutationImportanceAnalyzer(
        model=model,
        model_name=model_name,
        dataset_tag=dataset_tag,
    )

    analyzer.compute(X=artifacts["X_val_scaled"], y=artifacts["y_val"])
    importance_df = analyzer.get_importance_df()

    out_path = XAI_DIR / f"pfi_results_{model_name}_{dataset_tag}.csv"
    importance_df.to_csv(out_path, index=False)
    logger.info(f"[PFI] Resultados guardados en: {out_path.name}")

    return importance_df

def run_shap_local(artifacts: dict, model_name: str, group_file: Path, dataset_tag: str) -> pd.DataFrame:
    """Orquesta la ejecución de la explicabilidad local SHAP sobre un vecindario dado."""
    model = artifacts[f"{model_name}_model"]
    preprocessor = artifacts["preprocessor"]
    # feature_cols = CONTINUOUS_COLS + ORDINAL_COLS
    feature_cols = artifacts["X_val_raw"].columns.tolist()
    analyzer = ShapLocalAnalyzer(
        model=model,
        model_name=model_name,
        dataset_tag=dataset_tag
    )
    
    stability_df = analyzer.analyze_stability(
        neighborhood_path=group_file, #type: ignore
        feature_cols=feature_cols,
        preprocessor=preprocessor
    )
    return stability_df

if __name__ == "__main__":
    args = parse_args()

    if args.plot_stability:
        if not args.stability_file:
            logger.error("Debe proporcionar la ruta del CSV con --stability-file para poder graficar.")
            sys.exit(1)
            
        file_path = Path(args.stability_file)
        logger.info(f"Iniciando graficación de estabilidad para: {file_path.name}")
        
        LocalStabilityAnalyzer.plot_stability_scatter(file_path)
        logger.info("Operación finalizada.")
        sys.exit(0)

    tags = ["full", "clean"] if args.dataset == "both" else [args.dataset]
    model_names = ["dt", "xgb"] if args.model == "both" else [args.model]

    logger.info("=== Etapa XAI Iniciada ===")

    for tag in tags:
        artifacts = load_xai_artifacts(tag)

        if args.method == "intrinsic":
            for name in model_names:
                extract_intrinsic_importance(artifacts, name)

        if args.method in ("pfi", "all"):
            for name in model_names:
                run_pfi(artifacts, name)

        if args.method in ("surrogate", "all"):
            logger.info("[Surrogate] Módulo pendiente de implementación.")

        if args.method in ("lime", "all"):
            logger.info("[LIME] Módulo pendiente de implementación.")

        if args.method in ("shap", "all"):
                    if not args.group_file:
                        logger.warning(f"Ignorando SHAP local: No se proveyó el archivo de vecindario (--group-file).")
                    else:
                        group_path = Path(args.group_file)
                        for name in model_names:
                            run_shap_local(artifacts, name, group_path, tag)

    logger.info("=== Etapa XAI Completada ===")