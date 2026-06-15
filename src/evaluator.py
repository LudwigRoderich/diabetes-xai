"""
Evaluación de modelos de clasificación binaria y optimización de umbral.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from typing import Optional

from src.config import (
    DEFAULT_THRESHOLD, 
    THRESHOLD_SEARCH_MIN, 
    THRESHOLD_SEARCH_MAX, 
    THRESHOLD_STEPS,
    get_logger
)

logger = get_logger(__name__)


class Evaluator:
    """
    Calcula métricas de evaluación para modelos que exponen predict_proba(),
    y optimiza el umbral de decisión basado en distintas estrategias de negocio.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        logger.debug(f"Evaluator inicializado con umbral: {self.threshold:.3f}")

    def evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evalúa un modelo sobre el conjunto indicado y extrae métricas estáticas.
        """
        try:
            proba = self._get_proba(model, X)
            labels = self._proba_to_labels(proba, self.threshold)

            metrics = {
                "prauc": float(average_precision_score(y, proba)),
                "precision": float(precision_score(y, labels, zero_division=0)),
                "recall": float(recall_score(y, labels, zero_division=0)),
                "f1": float(f1_score(y, labels, zero_division=0)),
                "threshold": self.threshold,
            }
            logger.debug(f"Evaluación completada: PR-AUC={metrics['prauc']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Fallo durante la evaluación del modelo: {e}")
            raise

    def optimize_threshold(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "f1",
        min_recall: Optional[float] = None,
    ) -> float:
        """
        Busca el umbral óptimo barriendo el espacio de probabilidades
        según la estrategia matemática definida.
        """
        logger.info(f"Iniciando barrido de umbral. Estrategia: '{strategy}'")
        
        proba = self._get_proba(model, X)
        thresholds = np.linspace(THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX, THRESHOLD_STEPS)

        best_threshold = self.threshold
        best_score = -np.inf

        for t in thresholds:
            labels = self._proba_to_labels(proba, t)
            precision = float(precision_score(y, labels, zero_division=0))
            recall = float(recall_score(y, labels, zero_division=0))

            if min_recall is not None and recall < min_recall:
                continue

            score = self._compute_strategy_score(strategy, precision, recall)

            if score > best_score:
                best_score = score
                best_threshold = t

        if best_score == -np.inf:
            logger.warning(
                f"Ningún umbral cumple min_recall={min_recall}. "
                "Se relaja la restricción y se repite la búsqueda."
            )
            return self.optimize_threshold(model, X, y, strategy=strategy, min_recall=None)

        self.threshold = float(best_threshold)
        logger.info(f"Umbral óptimo encontrado: {self.threshold:.3f} (Score: {best_score:.4f})")
        return self.threshold

    def evaluate_curve(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """Devuelve los vectores de la curva Precision-Recall para gráficas."""
        proba = self._get_proba(model, X)
        precision_c, recall_c, thresh = precision_recall_curve(y, proba)

        return {
            "precision_curve": precision_c,
            "recall_curve": recall_c,
            "thresholds": thresh,
            "prauc": float(average_precision_score(y, proba)),
        }

    @staticmethod
    def _get_proba(model, X: pd.DataFrame) -> np.ndarray:
        """Extrae la probabilidad de la clase minoritaria (índice 1)."""
        if not hasattr(model, "predict_proba"):
            logger.error("El modelo proporcionado no expone el método predict_proba().")
            raise AttributeError("El modelo debe soportar predicción de probabilidades.")
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def _proba_to_labels(proba: np.ndarray, threshold: float) -> np.ndarray:
        """Discretiza probabilidades a etiquetas binarias."""
        return (proba >= threshold).astype(int)

    @staticmethod
    def _compute_strategy_score(strategy: str, precision: float, recall: float) -> float:
        """Calcula el score de acuerdo a la estrategia seleccionada."""
        if strategy == "f1":
            denom = precision + recall
            return (2 * precision * recall / denom) if denom > 0 else 0.0

        if strategy == "recall":
            return recall

        if strategy == "prauc":
            return precision * recall

        logger.error(f"Estrategia solicitada desconocida: {strategy}")
        raise ValueError(f"Estrategia '{strategy}' no reconocida.")