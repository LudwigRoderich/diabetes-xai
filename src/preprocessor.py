"""
Estandarización selectiva de variables según su tipo y contexto de uso.

Reglas:
  - Variables binarias  : nunca se escalan.
  - Variables continuas : se escalan siempre (entrenamiento y XAI).
  - Variables ordinales : se escalan únicamente para métodos XAI;
                          los modelos de clasificación las reciben sin escalar.

El scaler se ajusta (fit) exclusivamente sobre el fold de entrenamiento
y se aplica (transform) sobre los demás folds, nunca al revés.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional

from config import CONTINUOUS_COLS, ORDINAL_COLS, SCALERS_DIR


class Preprocessor:
    """
    Ajusta y aplica escalado estándar de forma selectiva según el contexto.

    Parameters
    ----------
    fold_id : int or str
        Identificador del fold o partición. Se usa para nombrar el archivo
        de serialización del scaler (ej. fold_0, final).
    dataset_tag : str
        Etiqueta del dataset usado: 'full' para el original, 'clean' para
        el depurado. Se incluye en el nombre del archivo guardado.

    Attributes
    ----------
    _scaler_model : StandardScaler
        Scaler ajustado sobre las variables continuas (para modelos).
    _scaler_xai : StandardScaler
        Scaler ajustado sobre continuas + ordinales (para XAI).
    _is_fitted : bool
        Indica si el objeto ya fue ajustado mediante fit().
    """

    def __init__(self, fold_id, dataset_tag: str = "full") -> None:
        self.fold_id     = fold_id
        self.dataset_tag = dataset_tag

        self._scaler_model = StandardScaler()
        self._scaler_xai   = StandardScaler()
        self._is_fitted    = False

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Ajusta ambos scalers sobre el conjunto de entrenamiento del fold.

        Parameters
        ----------
        X : pd.DataFrame
            Subconjunto de entrenamiento del fold actual. Debe contener
            todas las columnas definidas en CONTINUOUS_COLS y ORDINAL_COLS.

        Returns
        -------
        self
        """
        self._scaler_model.fit(X[CONTINUOUS_COLS])
        self._scaler_xai.fit(X[CONTINUOUS_COLS + ORDINAL_COLS])
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(
        self, X: pd.DataFrame, xai_mode: bool = False
    ) -> pd.DataFrame:
        """
        Aplica el escalado correspondiente sin modificar el DataFrame original.

        Parameters
        ----------
        X : pd.DataFrame
            Conjunto a transformar.
        xai_mode : bool
            Si False (defecto), escala únicamente CONTINUOUS_COLS.
            Si True, escala CONTINUOUS_COLS + ORDINAL_COLS.

        Returns
        -------
        pd.DataFrame
            Copia del DataFrame con las columnas correspondientes escaladas.
            Las columnas binarias permanecen intactas.
        """
        self._check_fitted()
        X_out = X.copy()

        if xai_mode:
            cols    = CONTINUOUS_COLS + ORDINAL_COLS
            scaler  = self._scaler_xai
        else:
            cols    = CONTINUOUS_COLS
            scaler  = self._scaler_model

        X_out[cols] = scaler.transform(X[cols])
        return X_out

    # ------------------------------------------------------------------
    def fit_transform(
        self, X: pd.DataFrame, xai_mode: bool = False
    ) -> pd.DataFrame:
        """
        Ajusta el scaler y transforma en un solo paso. Conveniente
        para el fold de entrenamiento.

        Parameters
        ----------
        X : pd.DataFrame
            Subconjunto de entrenamiento del fold.
        xai_mode : bool
            Ver transform().

        Returns
        -------
        pd.DataFrame
        """
        return self.fit(X).transform(X, xai_mode=xai_mode)

    # ------------------------------------------------------------------
    def save(self) -> Path:
        """
        Serializa ambos scalers en disco bajo SCALERS_DIR.

        El archivo se nombra: scaler_{dataset_tag}_fold{fold_id}.pkl

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        self._check_fitted()
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)

        path = SCALERS_DIR / f"scaler_{self.dataset_tag}_fold{self.fold_id}.pkl"
        joblib.dump(
            {
                "scaler_model": self._scaler_model,
                "scaler_xai"  : self._scaler_xai,
                "fold_id"     : self.fold_id,
                "dataset_tag" : self.dataset_tag,
            },
            path,
        )
        return path

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, fold_id, dataset_tag: str = "full") -> "Preprocessor":
        """
        Reconstituye un Preprocessor desde un archivo serializado.

        Parameters
        ----------
        fold_id : int or str
            Identificador del fold a recuperar.
        dataset_tag : str
            Etiqueta del dataset ('full' o 'clean').

        Returns
        -------
        Preprocessor
            Instancia con los scalers ya ajustados.
        """
        path = SCALERS_DIR / f"scaler_{dataset_tag}_fold{fold_id}.pkl"
        data = joblib.load(path)

        instance = cls(fold_id=data["fold_id"], dataset_tag=data["dataset_tag"])
        instance._scaler_model = data["scaler_model"]
        instance._scaler_xai   = data["scaler_xai"]
        instance._is_fitted    = True
        return instance

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Lanza un error si se intenta transformar antes de ajustar."""
        if not self._is_fitted:
            raise RuntimeError(
                "El Preprocessor no ha sido ajustado. Llama a fit() primero."
            )