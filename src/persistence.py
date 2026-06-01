"""
Serialización y recuperación de artefactos del proyecto.

Responsabilidades separadas por tipo de artefacto:
  - Modelos sklearn/XGBoost : joblib (.pkl)
  - Scalers (Preprocessor)  : gestionados por el propio Preprocessor
  - Métricas y resultados   : JSON (.json) y CSV (.csv)
  - Resúmenes tabulares     : CSV (.csv)

Todos los métodos son estáticos; la clase actúa como namespace.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any

from config import MODELS_DIR, RESULTS_DIR


class Persistence:
    """
    Colección de métodos estáticos para guardar y recuperar artefactos
    generados durante el entrenamiento y la evaluación.
    """

    # ==================================================================
    # Modelos
    # ==================================================================

    @staticmethod
    def save_model(model: Any, model_name: str, fold_id, dataset_tag: str) -> Path:
        """
        Serializa un modelo entrenado en disco.

        Parameters
        ----------
        model : objeto sklearn o XGBoost compatible con joblib.
        model_name : str
            Identificador del modelo, p.ej. 'dt' o 'xgb'.
        fold_id : int or str
            Identificador del fold; usar 'final' para el modelo definitivo.
        dataset_tag : str
            Etiqueta del dataset: 'full' o 'clean'.

        Returns
        -------
        Path : ruta del archivo guardado.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{model_name}_{dataset_tag}_fold{fold_id}.pkl"
        joblib.dump(model, path)
        return path

    @staticmethod
    def load_model(model_name: str, fold_id, dataset_tag: str) -> Any:
        """
        Recupera un modelo serializado desde disco.

        Parameters
        ----------
        model_name : str
            Identificador del modelo ('dt' o 'xgb').
        fold_id : int or str
            Identificador del fold.
        dataset_tag : str
            Etiqueta del dataset ('full' o 'clean').

        Returns
        -------
        Modelo deserializado.
        """
        path = MODELS_DIR / f"{model_name}_{dataset_tag}_fold{fold_id}.pkl"
        return joblib.load(path)

    # ==================================================================
    # Métricas por fold
    # ==================================================================

    @staticmethod
    def save_fold_metrics(metrics: dict, fold_id: int, dataset_tag: str) -> Path:
        """
        Guarda el diccionario de métricas de un fold como JSON.

        El diccionario debe tener la forma:
            {'dt': {prauc, precision, recall, threshold},
             'xgb': {prauc, precision, recall, threshold}}

        Parameters
        ----------
        metrics : dict
            Métricas devueltas por Evaluator.evaluate() para ambos modelos.
        fold_id : int
            Índice del fold.
        dataset_tag : str
            Etiqueta del dataset.

        Returns
        -------
        Path : ruta del archivo guardado.
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"metrics_{dataset_tag}_fold{fold_id}.json"

        payload = {
            "fold_id"    : fold_id,
            "dataset_tag": dataset_tag,
            "metrics"    : metrics,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load_fold_metrics(fold_id: int, dataset_tag: str) -> dict:
        """
        Recupera las métricas de un fold desde su archivo JSON.

        Returns
        -------
        dict con la estructura original guardada por save_fold_metrics().
        """
        path = RESULTS_DIR / f"metrics_{dataset_tag}_fold{fold_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================================================================
    # Resumen del CV (DataFrame agregado)
    # ==================================================================

    @staticmethod
    def save_cv_summary(summary: pd.DataFrame, dataset_tag: str) -> Path:
        """
        Guarda el DataFrame resumen del CV completo como CSV.

        Parameters
        ----------
        summary : pd.DataFrame
            Resultado de Trainer.get_cv_summary().
        dataset_tag : str
            Etiqueta del dataset.

        Returns
        -------
        Path : ruta del archivo guardado.
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"cv_summary_{dataset_tag}.csv"
        summary.to_csv(path, index=False)
        return path

    @staticmethod
    def load_cv_summary(dataset_tag: str) -> pd.DataFrame:
        """
        Recupera el DataFrame resumen del CV desde su CSV.

        Returns
        -------
        pd.DataFrame con las métricas agregadas por fold.
        """
        path = RESULTS_DIR / f"cv_summary_{dataset_tag}.csv"
        return pd.read_csv(path)

    # ==================================================================
    # Resultados del modelo final (threshold optimizado y métricas)
    # ==================================================================

    @staticmethod
    def save_final_results(results: dict, dataset_tag: str) -> Path:
        """
        Guarda los resultados del modelo final entrenado sobre X_train
        completo, incluyendo el umbral optimizado y las métricas sobre
        el conjunto de validación y tuning.

        Parameters
        ----------
        results : dict
            Estructura libre; se recomienda incluir al menos:
            {
              'dataset_tag'       : str,
              'optimal_threshold' : float,
              'val_metrics'       : {dt: {...}, xgb: {...}},
              'tune_metrics'      : {dt: {...}, xgb: {...}},
              'class_weights'     : {0: float, 1: float},
            }
        dataset_tag : str
            Etiqueta del dataset.

        Returns
        -------
        Path : ruta del archivo guardado.
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"final_results_{dataset_tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load_final_results(dataset_tag: str) -> dict:
        """
        Recupera los resultados finales desde su archivo JSON.

        Returns
        -------
        dict con la estructura guardada por save_final_results().
        """
        path = RESULTS_DIR / f"final_results_{dataset_tag}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================================================================
    # Utilidades generales
    # ==================================================================

    @staticmethod
    def list_saved_models(dataset_tag: str | None = None) -> list:
        """
        Lista los modelos serializados disponibles en MODELS_DIR.

        Parameters
        ----------
        dataset_tag : str, optional
            Si se indica, filtra por etiqueta de dataset.

        Returns
        -------
        list of Path : archivos .pkl encontrados.
        """
        if not MODELS_DIR.exists():
            return []
        pattern = f"*{dataset_tag}*.pkl" if dataset_tag else "*.pkl"
        return sorted(MODELS_DIR.glob(pattern))

    @staticmethod
    def list_saved_results(dataset_tag: str | None = None) -> list:
        """
        Lista los archivos de resultados disponibles en RESULTS_DIR.

        Parameters
        ----------
        dataset_tag : str, optional
            Si se indica, filtra por etiqueta de dataset.

        Returns
        -------
        list of Path : archivos .json y .csv encontrados.
        """
        if not RESULTS_DIR.exists():
            return []
        pattern = f"*{dataset_tag}*" if dataset_tag else "*"
        return sorted(
            p for p in RESULTS_DIR.glob(pattern)
            if p.suffix in {".json", ".csv"}
        )