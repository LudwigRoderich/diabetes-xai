"""
Punto de entrada de la etapa XAI del proyecto MM-700.

Carga los artefactos del pipeline de clasificación (modelos y preprocesador del
fold 'final') y orquesta los análisis de explicabilidad configurados por CLI.
"""
import argparse
import sys
import pandas as pd

from src.config import XAI_DIR, get_logger
from src.data_loader import DataLoader
from src.persistence import Persistence
from src.preprocessor import Preprocessor
from src.xai.pfi import PermutationImportanceAnalyzer

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
    return parser.parse_args()

def load_xai_artifacts(dataset_tag: str) -> dict:
    """Carga y estructura todos los artefactos necesarios para el análisis XAI."""
    use_clean = dataset_tag == "clean"
    logger.info(f"Cargando artefactos XAI | Dataset: {dataset_tag.upper()}")

    loader = DataLoader(use_clean=use_clean).load().split()
    prep = Preprocessor.load(fold_id="final", dataset_tag=dataset_tag)

    if loader.X_val is None or loader.y_val is None:
        logger.error(f"Datos de validación no disponibles para '{dataset_tag}'. Asegúrate de ejecutar el pipeline de clasificación primero.")
        sys.exit(1)

    X_val_raw = loader.X_val.copy()
    y_val = loader.y_val.copy()
    X_val_scaled = prep.transform(X_val_raw, xai_mode=False)

    models = {}
    for name in ["dt", "xgb"]:
        try:
            models[name] = Persistence.load_model(name, "final", dataset_tag)
            logger.debug(f"Modelo {name.upper()} ({dataset_tag}) cargado.")
        except FileNotFoundError:
            logger.warning(f"Modelo {name.upper()} no encontrado para dataset '{dataset_tag}'. Se omitirá.")

    return {
        "loader": loader,
        "prep": prep,
        "X_val_raw": X_val_raw,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val,
        "models": models,
        "dataset_tag": dataset_tag,
    }

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

if __name__ == "__main__":
    args = parse_args()
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
            logger.info("[SHAP] Módulo pendiente de implementación.")

    logger.info("=== Etapa XAI Completada ===")