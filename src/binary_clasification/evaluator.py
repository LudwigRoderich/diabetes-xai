"""
Evaluación de modelos de clasificación binaria y optimización de umbral.
"""
import inspect
from typing import Optional, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef
)

from src.config import (
    DEFAULT_THRESHOLD,
    get_logger
)

logger = get_logger(__name__)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """División elemento a elemento que devuelve 0.0 donde el denominador es 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _precision(tp, fp, fn, tn):
    return _safe_div(tp, tp + fp)


def _recall(tp, fp, fn, tn):
    return _safe_div(tp, tp + fn)


def _specificity(tp, fp, fn, tn):
    return _safe_div(tn, tn + fp)


def _fbeta(tp, fp, fn, tn, beta: float):
    p = _precision(tp, fp, fn, tn)
    r = _recall(tp, fp, fn, tn)
    b2 = beta ** 2
    return _safe_div((1 + b2) * p * r, b2 * p + r)


def _score_precision(tp, fp, fn, tn):
    return _precision(tp, fp, fn, tn)


def _score_recall(tp, fp, fn, tn):
    return _recall(tp, fp, fn, tn)


def _score_f1(tp, fp, fn, tn):
    return _fbeta(tp, fp, fn, tn, beta=1.0)


def _score_f2(tp, fp, fn, tn):
    """F-beta con beta=2: pondera el recall el doble que la precisión.
    Útil cuando el costo de un falso negativo (no detectar la clase
    minoritaria de interés) es mayor que el de un falso positivo."""
    return _fbeta(tp, fp, fn, tn, beta=2.0)


def _score_f05(tp, fp, fn, tn):
    """F-beta con beta=0.5: pondera la precisión el doble que el recall."""
    return _fbeta(tp, fp, fn, tn, beta=0.5)


def _score_balanced_accuracy(tp, fp, fn, tn):
    """Promedio de sensibilidad (recall) y especificidad. Cada clase
    pesa igual sin importar su tamaño, útil bajo desbalance fuerte."""
    return (_recall(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2.0


def _score_youden_j(tp, fp, fn, tn):
    """Estadístico J de Youden: sensibilidad + especificidad - 1.
    Clásico en literatura biomédica para elegir umbral con clases
    desbalanceadas; rango [-1, 1], 0 equivale a un clasificador aleatorio."""
    return _recall(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn) - 1.0


def _score_gmean(tp, fp, fn, tn):
    """Media geométrica de sensibilidad y especificidad. A diferencia del
    promedio simple, castiga con más fuerza si una de las dos es cercana
    a 0 (evita umbrales que sacrifican por completo una clase)."""
    product = _recall(tp, fp, fn, tn) * _specificity(tp, fp, fn, tn)
    return np.sqrt(np.clip(product, 0, None))


def _score_mcc(tp, fp, fn, tn):
    """Coeficiente de correlación de Matthews. Considera las 4 celdas de
    la matriz de confusión simultáneamente; se considera la métrica más
    informativa en un único número para desbalance severo (rango [-1, 1])."""
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_div(num, den)


class Evaluator:
    """
    Calcula métricas de evaluación para modelos que exponen predict_proba(),
    y optimiza el umbral de decisión basado en distintas estrategias de negocio.
    """

    # Registro de estrategias disponibles para optimize_threshold(strategy=...).
    # Cada función recibe (tp, fp, fn, tn) como arreglos numpy y devuelve un
    # arreglo de scores del mismo tamaño. Extensible en runtime vía
    # Evaluator.register_strategy(nombre, funcion) sin modificar este archivo.
    _STRATEGIES: dict[str, Callable] = {
        "precision": _score_precision,
        "recall": _score_recall,
        "f1": _score_f1,
        "f2": _score_f2,
        "f0.5": _score_f05,
        "balanced_accuracy": _score_balanced_accuracy,
        "youden_j": _score_youden_j,
        "gmean": _score_gmean,
        "mcc": _score_mcc,
    }

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        logger.debug(f"Evaluator inicializado con umbral: {self.threshold:.3f}")

    @classmethod
    def register_strategy(cls, name: str, func: Callable) -> None:
        """
        Registra una estrategia adicional para optimize_threshold(strategy=name).

        `func` debe aceptar (tp, fp, fn, tn) como arreglos numpy y devolver
        un arreglo de scores del mismo tamaño (vectorizado, sin loops sobre
        umbrales individuales).
        """
        if name == "prauc":
            raise ValueError(
                f"'{name}' está reservado y no puede registrarse: el PR-AUC "
                "no es una función de un único umbral (ver evaluate_curve())."
            )
        cls._STRATEGIES[name] = func
        logger.info(f"Estrategia '{name}' registrada en Evaluator.")

    def evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evalúa un modelo sobre el conjunto indicado y extrae métricas estáticas.
        """
        try:
            if not np.all(np.isin(y, [0, 1])):
                logger.error("Las etiquetas verdaderas (y) deben ser binarias (0 o 1).")
                raise ValueError("Las etiquetas verdaderas (y) deben ser binarias (0 o 1).")
            
            proba = self._get_proba(model, X)
            labels = self._proba_to_labels(proba, self.threshold)
            

            metrics = {
                "prauc": float(average_precision_score(y, proba)),
                "precision": float(precision_score(y, labels, zero_division=0)),
                "recall": float(recall_score(y, labels, zero_division=0)),
                "f1": float(f1_score(y, labels, zero_division=0)),
                "f2": float(fbeta_score(y, labels, beta=2.0, zero_division=0)),
                "mcc": float(matthews_corrcoef(y, labels)),
                "threshold": self.threshold,
            }
            logger.debug(f"Evaluación completada: Métricas={metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Fallo durante la evaluación del modelo: {e}")
            raise

    def optimize_threshold(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str | Callable = "f1",
        min_recall: Optional[float] = None,
    ) -> float:
        """
        Busca el umbral óptimo evaluando únicamente los puntos de cambio
        exactos en la distribución empírica de probabilidades, mediante
        una curva de confusión acumulada (O(n log n) en tiempo, O(n) en
        memoria).

        `strategy` puede ser:
          - Un string registrado en Evaluator._STRATEGIES: "precision",
            "recall", "f1", "f2", "f0.5", "balanced_accuracy", "youden_j",
            "gmean", "mcc".
          - Un Callable con firma (tp, fp, fn, tn) -> array, evaluado de
            forma vectorizada sobre todos los umbrales a la vez.
          - Un Callable con firma antigua (y_true, y_pred) -> float, que
            fuerza un fallback no vectorizado (lento, O(n * |umbrales|)
            en tiempo/memoria); se emite una advertencia en ese caso.
        """
        logger.info(f"Iniciando optimización vectorizada de umbral (estrategia={strategy!r}).")

        proba = self._get_proba(model, X)
        if not np.all(np.isin(y, [0, 1])):
            logger.error("Las etiquetas verdaderas (y) deben ser binarias (0 o 1).")
            raise ValueError("Las etiquetas verdaderas (y) deben ser binarias (0 o 1).")
        
        y_true = np.asarray(y)

        tp, fp, fn, tn, thresholds = self._binary_confusion_curve(y_true, proba)
        recall = _recall(tp, fp, fn, tn)

        if callable(strategy):
            n_params = len(inspect.signature(strategy).parameters)
            if n_params == 4:
                scores = np.asarray(strategy(tp, fp, fn, tn), dtype=float)
            else:
                logger.warning(
                    "El callable 'strategy' usa la firma antigua (y_true, y_pred). "
                    "Se recurre a un fallback no vectorizado que puede ser lento "
                    "y consumir mucha memoria en datasets grandes. Se recomienda "
                    "reescribir 'strategy' con firma (tp, fp, fn, tn) -> array."
                )
                scores = np.array([
                    float(strategy(y_true, self._proba_to_labels(proba, t)))
                    for t in thresholds
                ])
        else:
            scores = self._compute_strategy_score(strategy, tp, fp, fn, tn)
            scores = np.asarray(scores, dtype=float)

        valid = np.ones_like(scores, dtype=bool)
        if min_recall is not None:
            valid &= recall >= min_recall
            if not valid.any():
                logger.warning(
                    f"Ningún umbral cumple min_recall={min_recall}. "
                    "Se relaja la restricción y se repite la búsqueda."
                )
                return self.optimize_threshold(model, X, y, strategy=strategy, min_recall=None)

        scores_masked = np.where(valid, scores, -np.inf)
        best_idx = int(np.argmax(scores_masked))
        best_score = float(scores_masked[best_idx])

        self.threshold = float(thresholds[best_idx])
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
    def _binary_confusion_curve(
        y_true: np.ndarray, proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Réplica del algoritmo interno de sklearn (_binary_clf_curve):
        calcula TP, FP, FN, TN para cada umbral distinto en una sola
        pasada ordenada, sin iterar sobre las observaciones por umbral.
        """
        order = np.argsort(proba, kind="mergesort")[::-1]
        y_sorted = y_true[order]
        proba_sorted = proba[order]

        distinct_idx = np.where(np.diff(proba_sorted))[0]
        threshold_idxs = np.r_[distinct_idx, y_sorted.size - 1]

        tp = np.cumsum(y_sorted)[threshold_idxs].astype(float)
        fp = (threshold_idxs + 1 - tp)

        P = tp[-1]
        N = fp[-1]

        fn = P - tp
        tn = N - fp
        thresholds = proba_sorted[threshold_idxs]

        return tp, fp, fn, tn, thresholds

    @classmethod
    def _compute_strategy_score(
        cls,
        strategy: str,
        tp: np.ndarray,
        fp: np.ndarray,
        fn: np.ndarray,
        tn: np.ndarray,
    ) -> np.ndarray:
        """Despacha el cálculo de score a la función registrada en _STRATEGIES."""
        if strategy == "prauc":
            logger.error(
                "'prauc' no es una estrategia válida para optimize_threshold: "
                "PR-AUC es una integral sobre todos los umbrales, no una "
                "función de un único umbral. Usa evaluate_curve() + "
                "average_precision_score para obtener el PR-AUC real."
            )
            raise ValueError(
                "Estrategia 'prauc' no soportada en _compute_strategy_score: "
                "el PR-AUC es independiente del umbral y no puede maximizarse "
                "punto a punto. Usa evaluate_curve()."
            )

        func = cls._STRATEGIES.get(strategy)
        if func is None:
            logger.error(f"Estrategia solicitada desconocida: {strategy}")
            raise ValueError(
                f"Estrategia '{strategy}' no reconocida. Disponibles: "
                f"{sorted(cls._STRATEGIES.keys())}. También puedes registrar "
                "una nueva con Evaluator.register_strategy(nombre, funcion)."
            )
        return func(tp, fp, fn, tn)
