"""
Gestión de carga y partición estratificada del dataset.
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from src.config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH,
    TARGET_COL,
    TRAIN_SIZE, VAL_SIZE, TUNE_SIZE,
    N_FOLDS,
    SPLIT_SEED, CV_SEED,
    get_logger
)

logger = get_logger(__name__)

class DataLoader:
    """
    Carga el dataset de diabetes y gestiona sus particiones (Train, Val, Tune, Folds)
    garantizando la representatividad estadística (estratificación).
    """

    def __init__(self, use_clean: bool = False) -> None:
        self.use_clean = use_clean
        self.df        = None

        self.X_train = self.X_val = self.X_tune = None
        self.y_train = self.y_val = self.y_tune = None

    def load(self) -> "DataLoader":
        """Carga el dataset desde disco verificando su existencia."""
        path = CLEAN_DATA_PATH if self.use_clean else RAW_DATA_PATH
        
        if not path.exists():
            logger.error(f"El archivo no existe en la ruta especificada: {path}")
            raise FileNotFoundError(f"Dataset no encontrado: {path}")

        logger.info(f"Cargando dataset {'limpio' if self.use_clean else 'completo'}: {path.name}")
        self.df = pd.read_csv(path)
        logger.debug(f"Dataset cargado exitosamente. Dimensiones: {self.df.shape}")
        
        return self

    def split(self) -> "DataLoader":
        """
        Realiza la partición estratificada respetando TRAIN_SIZE, VAL_SIZE y TUNE_SIZE.
        """
        if self.df is None:
            logger.error("Intento de ejecutar split() sin dataset cargado.")
            raise RuntimeError("Llama a load() antes de split().")

        logger.info("Iniciando partición estratificada (Train, Val, Tune)...")
        
        X = self.df.drop(columns=[TARGET_COL])
        y = self.df[TARGET_COL]

        # 1. Extraer conjunto de Tuning (10%)
        sss_tune = StratifiedShuffleSplit(n_splits=1, test_size=TUNE_SIZE, random_state=SPLIT_SEED)
        idx_main, idx_tune = next(sss_tune.split(X, y))
        X_main, y_main = X.iloc[idx_main], y.iloc[idx_main]
        self.X_tune, self.y_tune = X.iloc[idx_tune], y.iloc[idx_tune]

        # 2. Separar Train (70%) y Val (20%) del 90% restante
        relative_val = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=SPLIT_SEED)
        idx_train, idx_val = next(sss_val.split(X_main, y_main))
        
        self.X_train, self.y_train = X_main.iloc[idx_train], y_main.iloc[idx_train]
        self.X_val, self.y_val     = X_main.iloc[idx_val], y_main.iloc[idx_val]

        logger.debug(
            f"Partición completada | "
            f"Train: {len(self.X_train)} ({TRAIN_SIZE*100:.0f}%), "
            f"Val: {len(self.X_val)} ({VAL_SIZE*100:.0f}%), "
            f"Tune: {len(self.X_tune)} ({TUNE_SIZE*100:.0f}%)"
        )
        return self

    def get_cv_folds(self) -> list:
        """Genera los índices de los pliegues para validación cruzada."""
        if self.X_train is None or self.y_train is None:
            logger.error("Intento de generar folds sin particiones de entrenamiento.")
            raise RuntimeError("Llama a split() antes de get_cv_folds().")

        logger.info(f"Generando {N_FOLDS} pliegues estratificados para CV.")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        return list(skf.split(self.X_train, self.y_train))

    def class_proportions(self, y: pd.Series | None = None) -> dict:
        """Devuelve la distribución porcentual de las clases objetivo."""
        target = y if y is not None else self.y_train
        if target is None:
            logger.error("No hay etiquetas disponibles para calcular proporciones.")
            raise RuntimeError("No hay etiquetas disponibles.")
            
        proportions = target.value_counts(normalize=True).to_dict()
        logger.debug(f"Proporciones de clase calculadas: {proportions}")
        return proportions