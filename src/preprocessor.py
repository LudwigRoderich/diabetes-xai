"""
Módulo para el preprocesamiento y escalado selectivo de variables.
Responsable de manejar las transformaciones necesarias para modelos y XAI.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.config import CONTINUOUS_COLS, ORDINAL_COLS, SCALERS_DIR, get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Gestiona el escalado estándar de variables con lógica selectiva.
    """
    def __init__(self, fold_id: str | int, dataset_tag: str = "full") -> None:
        self.fold_id = fold_id
        self.dataset_tag = dataset_tag
        
        self._scaler_model = StandardScaler()
        self._scaler_xai = StandardScaler()
        self._is_fitted = False
        
        logger.debug(f"Preprocessor inicializado -> fold: {fold_id} | dataset: {dataset_tag}")

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Ajusta los escaladores sobre el conjunto de entrenamiento."""
        try:
            # 1. Ajuste para los modelos predictivos (solo continuas)
            self._scaler_model.fit(X[CONTINUOUS_COLS])
            
            # 2. Ajuste estricto para XAI (continuas + ordinales)
            xai_cols = CONTINUOUS_COLS + ORDINAL_COLS
            self._scaler_xai.fit(X[xai_cols])
            
            self._is_fitted = True
            logger.debug(f"Escaladores ajustados exitosamente para el fold {self.fold_id}.")
        except KeyError as e:
            logger.error(f"Error al acceder a las columnas durante fit() en fold {self.fold_id}: {e}")
            raise
            
        return self

    def transform(self, X: pd.DataFrame, xai_mode: bool = False) -> pd.DataFrame:
        """Aplica la transformación seleccionada sin modificar el DataFrame original."""
        self._check_fitted()
        X_out = X.copy()
        
        try:
            if xai_mode:
                cols = CONTINUOUS_COLS + ORDINAL_COLS
                X_out[cols] = self._scaler_xai.transform(X[cols])
            else:
                cols = CONTINUOUS_COLS
                X_out[cols] = self._scaler_model.transform(X[cols])
        except KeyError as e:
            logger.error(f"Faltan columnas esperadas durante transform() en fold {self.fold_id}: {e}")
            raise
            
        return X_out

    def fit_transform(self, X: pd.DataFrame, xai_mode: bool = False) -> pd.DataFrame:
        """Ajusta el scaler y transforma en un solo paso."""
        return self.fit(X).transform(X, xai_mode=xai_mode)

    def save(self) -> Path:
        """Serializa el preprocessor actual."""
        self._check_fitted()
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)
        
        path = SCALERS_DIR / f"scaler_{self.dataset_tag}_fold{self.fold_id}.pkl"
        
        joblib.dump({
            "scaler_model": self._scaler_model,
            "scaler_xai": self._scaler_xai,
            "fold_id": self.fold_id,
            "dataset_tag": self.dataset_tag
        }, path)
        
        logger.debug(f"Preprocessor guardado en: {path.name}")
        return path

    @classmethod
    def load(cls, fold_id: str | int, dataset_tag: str = "full") -> "Preprocessor":
        """Reconstituye un Preprocessor desde un archivo serializado."""
        path = SCALERS_DIR / f"scaler_{dataset_tag}_fold{fold_id}.pkl"
        
        if not path.exists():
            logger.error(f"No se encontró el scaler serializado en: {path}")
            raise FileNotFoundError(f"Scaler no encontrado: {path}")
            
        data = joblib.load(path)
        
        instance = cls(fold_id=data["fold_id"], dataset_tag=data["dataset_tag"])
        instance._scaler_model = data["scaler_model"]
        instance._scaler_xai = data["scaler_xai"]
        instance._is_fitted = True
        
        logger.info(f"Preprocessor cargado desde disco: {path.name}")
        return instance

    def _check_fitted(self) -> None:
        """Lanza un error si se intenta transformar antes de ajustar."""
        if not self._is_fitted:
            logger.error(f"Intento de usar transform() en Preprocessor sin entrenar (fold: {self.fold_id}).")
            raise RuntimeError("El Preprocessor no ha sido ajustado. Llama a fit() primero.")