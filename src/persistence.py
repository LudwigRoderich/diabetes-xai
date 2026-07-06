"""
Serialización y recuperación de artefactos del proyecto.
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any

from src.config import MODELS_DIR, RESULTS_CLF_DIR, get_logger

logger = get_logger(__name__)


class Persistence:
    """
    Colección de métodos estáticos para guardar y recuperar artefactos
    generados durante el entrenamiento y la evaluación.
    """

    @staticmethod
    def save_model(model: Any, model_name: str, fold_id: int | str, dataset_tag: str) -> Path:
        """Serializa un modelo entrenado en disco."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{model_name}_{dataset_tag}_fold{fold_id}.pkl"
        
        try:
            joblib.dump(model, path)
            logger.debug(f"Modelo guardado: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Error al guardar modelo {model_name} en fold {fold_id}: {e}")
            raise

    @staticmethod
    def load_model(model_name: str, fold_id: int | str, dataset_tag: str) -> Any:
        """Recupera un modelo serializado desde disco."""
        path = MODELS_DIR / f"{model_name}_{dataset_tag}_fold{fold_id}.pkl"
        
        if not path.exists():
            logger.error(f"No se encontró el modelo: {path}")
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
            
        model = joblib.load(path)
        logger.debug(f"Modelo cargado: {path.name}")
        return model

    @staticmethod
    def save_fold_metrics(metrics: dict, fold_id: int | str, dataset_tag: str) -> Path:
        """Guarda el diccionario de métricas de un fold como JSON."""
        path = RESULTS_CLF_DIR / f"metrics_{dataset_tag}_fold{fold_id}.json"

        payload = {
            "fold_id": fold_id,
            "dataset_tag": dataset_tag,
            "metrics": metrics,
        }
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug(f"Métricas de fold guardadas: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Error al guardar métricas del fold {fold_id}: {e}")
            raise

    @staticmethod
    def load_fold_metrics(fold_id: int | str, dataset_tag: str) -> dict:
        """Recupera las métricas de un fold desde su archivo JSON."""
        path = RESULTS_CLF_DIR / f"metrics_{dataset_tag}_fold{fold_id}.json"
        
        if not path.exists():
            logger.error(f"No se encontraron métricas de fold en: {path}")
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.debug(f"Métricas de fold cargadas: {path.name}")
        return data

    @staticmethod
    def save_cv_summary(summary: pd.DataFrame, dataset_tag: str) -> Path:
        """Guarda el DataFrame resumen del CV completo como CSV."""
        RESULTS_CLF_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_CLF_DIR / f"cv_summary_{dataset_tag}.csv"
        
        try:
            summary.to_csv(path, index=False)
            logger.info(f"Resumen CV guardado: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Error al guardar resumen CV para dataset '{dataset_tag}': {e}")
            raise

    @staticmethod
    def load_cv_summary(dataset_tag: str) -> pd.DataFrame:
        """Recupera el DataFrame resumen del CV desde su CSV."""
        path = RESULTS_CLF_DIR / f"cv_summary_{dataset_tag}.csv"
        
        if not path.exists():
            logger.error(f"No se encontró el resumen CV: {path}")
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
            
        df = pd.read_csv(path)
        logger.info(f"Resumen CV cargado: {path.name}")
        return df

    @staticmethod
    def save_final_results(results: dict, dataset_tag: str) -> Path:
        """Guarda los resultados del modelo final entrenado sobre X_train completo."""
        RESULTS_CLF_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_CLF_DIR / f"final_results_{dataset_tag}.json"
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Resultados finales guardados: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Error al guardar resultados finales para dataset '{dataset_tag}': {e}")
            raise

    @staticmethod
    def load_final_results(dataset_tag: str) -> dict:
        """Recupera los resultados finales desde su archivo JSON."""
        path = RESULTS_CLF_DIR / f"final_results_{dataset_tag}.json"
        
        if not path.exists():
            logger.error(f"No se encontraron los resultados finales: {path}")
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"Resultados finales cargados: {path.name}")
        return data

    @staticmethod
    def list_saved_models(dataset_tag: str | None = None) -> list:
        """Lista los modelos serializados disponibles en MODELS_DIR."""
        if not MODELS_DIR.exists():
            logger.warning(f"El directorio de modelos no existe: {MODELS_DIR}")
            return []
            
        pattern = f"*{dataset_tag}*.pkl" if dataset_tag else "*.pkl"
        models_found = sorted(MODELS_DIR.glob(pattern))
        logger.debug(f"Se encontraron {len(models_found)} modelos guardados.")
        return models_found

    @staticmethod
    def list_saved_results(dataset_tag: str | None = None) -> list:
        """Lista los archivos de resultados disponibles en RESULTS_CLF_DIR."""
        if not RESULTS_CLF_DIR.exists():
            logger.warning(f"El directorio de resultados no existe: {RESULTS_CLF_DIR}")
            return []
            
        pattern = f"*{dataset_tag}*" if dataset_tag else "*"
        results_found = sorted(
            p for p in RESULTS_CLF_DIR.glob(pattern)
            if p.suffix in {".json", ".csv"}
        )
        logger.debug(f"Se encontraron {len(results_found)} archivos de resultados.")
        return results_found