"""
Inferencia centralizada sobre modelos ya entrenados.

Punto único para recuperar, para cualquier combinación de
(modelo, dataset, partición), la predicción o probabilidad asignada,
aplicando siempre el preprocesado (scaler) aprendido en entrenamiento
(nunca re-ajustado sobre la partición consultada).
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DEFAULT_THRESHOLD, PARTITION_TAGS, get_logger
from .data_loader import DataLoader
from .persistence import Persistence
from .preprocessor import Preprocessor
from .visualizer import plot_confusion_matrix

logger = get_logger(__name__)


class ModelInference:
    """
    Recupera modelo + preprocesador + partición desde disco de forma
    determinista.

    Cachea loaders/preprocesadores/modelos por instancia para no releer
    disco en cada llamada dentro de una misma ejecución.
    """

    def __init__(self) -> None:
        self._loader_cache: dict[str, DataLoader] = {}
        self._prep_cache: dict[str, Preprocessor] = {}
        self._model_cache: dict[tuple[str, str], object] = {}

    def _get_loader(self, dataset_tag: str) -> DataLoader:
        if dataset_tag not in self._loader_cache:
            logger.debug(f"Reconstruyendo particiones de '{dataset_tag}' (determinista vía SPLIT_SEED).")
            self._loader_cache[dataset_tag] = DataLoader(use_clean=(dataset_tag == "clean")).load().split()
        return self._loader_cache[dataset_tag]

    def _get_preprocessor(self, dataset_tag: str) -> Preprocessor:
        if dataset_tag not in self._prep_cache:
            self._prep_cache[dataset_tag] = Preprocessor.load(fold_id="final", dataset_tag=dataset_tag)
        return self._prep_cache[dataset_tag]

    def _get_model(self, model_tag: str, dataset_tag: str):
        key = (model_tag, dataset_tag)
        if key not in self._model_cache:
            self._model_cache[key] = Persistence.load_model(
                model_name=model_tag, fold_id="final", dataset_tag=dataset_tag
            )
        return self._model_cache[key]

    def get_partition(self, dataset_tag: str, partition: str) -> tuple[pd.DataFrame, pd.Series]:
        """Devuelve (X_raw, y_true) de la partición pedida, sin escalar."""
        if partition not in PARTITION_TAGS:
            raise ValueError(f"Partición desconocida: '{partition}'. Usa una de: {PARTITION_TAGS}")

        loader = self._get_loader(dataset_tag)
        partition_map = {
            "train": (loader.X_train, loader.y_train),
            "val": (loader.X_val, loader.y_val),
            "tune": (loader.X_tune, loader.y_tune),
        }
        if partition not in partition_map:
            raise ValueError(f"Partición desconocida: '{partition}'. Usa una de: {list(partition_map.keys())}")
        if partition_map[partition][0] is None or partition_map[partition][1] is None:
            raise RuntimeError(f"Partición '{partition}' no disponible en dataset '{dataset_tag}'.")
        return partition_map[partition] #type: ignore

    def predict(
        self,
        dataset_tag: str,
        model_tag: str,
        partition: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Devuelve {"y_true", "y_pred", "proba"} para (model_tag, dataset_tag,
        partition), recuperando modelo y scaler de disco y aplicando el
        preprocesado de entrenamiento sobre la partición solicitada.

        Si el modelo no expone predict_proba() (no todos los estimadores
        lo hacen), se usa predict() directo y el umbral se ignora —
        se emite advertencia en ese caso.
        """
        X_raw, y_true = self.get_partition(dataset_tag, partition)
        prep = self._get_preprocessor(dataset_tag)
        model = self._get_model(model_tag, dataset_tag)

        X_scaled = prep.transform(X_raw)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[:, 1] #type: ignore
            y_pred = (proba >= threshold).astype(int)
        else:
            if not hasattr(model, "predict"):
                raise AttributeError(f"El modelo '{model_tag}' no expone ni predict_proba() ni predict().")
            logger.warning(
                f"'{model_tag}' no expone predict_proba(); se usa predict() directo "
                "y el umbral solicitado no aplica."
            )
            proba = None
            y_pred = model.predict(X_scaled) #type: ignore

        return {"y_true": y_true, "y_pred": y_pred, "proba": proba}

    def resolve_threshold(
        self,
        model_tag: str,
        dataset_tag: str,
        threshold: Optional[float] = None,
        thresh_metric: Optional[str] = None,
    ) -> float:
        """
        Resuelve qué umbral usar, en orden de prioridad:
          1. `threshold` explícito (override manual, para pruebas puntuales).
          2. Umbral óptimo persistido para (model_tag, dataset_tag,
             thresh_metric), si ese experimento existe en disco.
          3. DEFAULT_THRESHOLD (0.5), con advertencia — no es un umbral
             "óptimo", es el valor por defecto sin optimizar.
        """
        if threshold is not None:
            logger.info(f"[{dataset_tag}/{model_tag}] Umbral manual={threshold:.3f} (override explícito).")
            return float(threshold)

        if thresh_metric is not None:
            try:
                final_results = Persistence.load_final_results(dataset_tag, thresh_metric)
                optimal = final_results.get("optimal_thresholds", {}).get(model_tag)
                if optimal is not None:
                    logger.info(
                        f"[{dataset_tag}/{model_tag}] Umbral óptimo recuperado "
                        f"(estrategia='{thresh_metric}'): {optimal:.3f}"
                    )
                    return float(optimal)
                logger.warning(
                    f"El experimento '{thresh_metric}' para '{dataset_tag}' no tiene "
                    f"umbral óptimo guardado para '{model_tag}'."
                )
            except FileNotFoundError:
                logger.warning(
                    f"No existe un experimento persistido para dataset='{dataset_tag}' "
                    f"con estrategia='{thresh_metric}'."
                )

        logger.warning(
            f"[{dataset_tag}/{model_tag}] Usando DEFAULT_THRESHOLD={DEFAULT_THRESHOLD}: "
            "no se encontró umbral óptimo ni se indicó uno manual."
        )
        return DEFAULT_THRESHOLD

    def confusion_matrix(
        self,
        dataset_tag: str,
        model_tag: str,
        partition: str,
        threshold: Optional[float] = None,
        thresh_metric: Optional[str] = None,
    ) -> Path:
        """
        Genera y guarda la matriz de confusión para (model_tag, dataset_tag,
        partition), resolviendo el umbral automáticamente (óptimo persistido
        o manual si se indica).
        """
        resolved_threshold = self.resolve_threshold(model_tag, dataset_tag, threshold, thresh_metric)
        result = self.predict(dataset_tag, model_tag, partition, threshold=resolved_threshold)

        out_path = plot_confusion_matrix(
            y_true=result["y_true"],
            y_pred=result["y_pred"],
            model_tag=model_tag,
            dataset_tag=dataset_tag,
            partition=partition,
            threshold=resolved_threshold,
        )
        logger.info(
            f"[{dataset_tag}/{model_tag}] Matriz de confusión ({partition}, "
            f"umbral={resolved_threshold:.3f}) guardada: {out_path.name}"
        )
        return out_path
