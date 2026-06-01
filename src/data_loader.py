"""
Carga del dataset y partición estratificada en tres conjuntos:
entrenamiento (70 %), validación (20 %) y tuning (10 %).
Genera además los k-folds estratificados sobre el conjunto de entrenamiento.
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH,
    TARGET_COL,
    TRAIN_SIZE, VAL_SIZE, TUNE_SIZE,
    N_FOLDS,
    SPLIT_SEED, CV_SEED,
)


class DataLoader:
    """
    Gestiona la carga y partición estratificada del dataset de diabetes.

    Parameters
    ----------
    use_clean : bool
        Si True, carga el dataset sin outliers (diabetes_clean.csv);
        si False, carga el dataset original completo.

    Attributes
    ----------
    df : pd.DataFrame
        Dataset completo cargado desde disco.
    X_train, X_val, X_tune : pd.DataFrame
        Conjuntos de características para entrenamiento, validación y tuning.
    y_train, y_val, y_tune : pd.Series
        Etiquetas correspondientes a cada partición.
    """

    def __init__(self, use_clean: bool = False) -> None:
        self.use_clean = use_clean
        self.df        = None

        self.X_train = self.X_val = self.X_tune = None
        self.y_train = self.y_val = self.y_tune = None

    # ------------------------------------------------------------------
    def load(self) -> "DataLoader":
        """Carga el CSV desde la ruta configurada en config.py."""
        path = CLEAN_DATA_PATH if self.use_clean else RAW_DATA_PATH
        self.df = pd.read_csv(path)
        return self

    # ------------------------------------------------------------------
    def split(self) -> "DataLoader":
        """
        Particiona el dataset en train / val / tune respetando la
        proporción de la variable objetivo en cada subconjunto.

        El proceso es:
          1. Separar TUNE  (10 %) del total.
          2. Del 90 % restante, separar VAL (≈22.2 % → 20 % del total).
          3. El resto constituye TRAIN (≈70 % del total).
        """
        if self.df is None:
            raise RuntimeError("Llama a load() antes de split().")

        X = self.df.drop(columns=[TARGET_COL])
        y = self.df[TARGET_COL]

        # --- paso 1: separar tuning ----------------------------------
        sss_tune = StratifiedShuffleSplit(
            n_splits=1,
            test_size=TUNE_SIZE,
            random_state=SPLIT_SEED,
        )
        idx_main, idx_tune = next(sss_tune.split(X, y))
        X_main, y_main = X.iloc[idx_main], y.iloc[idx_main]
        self.X_tune, self.y_tune = X.iloc[idx_tune], y.iloc[idx_tune]

        # --- paso 2: separar validación del 90 % restante -----------
        # VAL_SIZE / (TRAIN_SIZE + VAL_SIZE) = 0.20 / 0.90 ≈ 0.2222
        relative_val = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_val,
            random_state=SPLIT_SEED,
        )
        idx_train, idx_val = next(sss_val.split(X_main, y_main))
        self.X_train, self.y_train = X_main.iloc[idx_train], y_main.iloc[idx_train]
        self.X_val,   self.y_val   = X_main.iloc[idx_val],   y_main.iloc[idx_val]

        return self

    # ------------------------------------------------------------------
    def get_cv_folds(self):
        """
        Genera los índices de los k-folds estratificados sobre
        el conjunto de entrenamiento.

        Returns
        -------
        list of (ndarray, ndarray)
            Lista de tuplas (idx_train_fold, idx_val_fold) con índices
            posicionales sobre X_train / y_train.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Llama a split() antes de get_cv_folds().")

        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=CV_SEED,
        )
        return list(skf.split(self.X_train, self.y_train))

    # ------------------------------------------------------------------
    def class_proportions(self, y: pd.Series | None = None) -> dict:
        """
        Calcula la proporción de cada clase en la serie indicada.
        Si no se pasa ninguna, usa y_train.

        Returns
        -------
        dict : {clase: proporción}
        """
        target = y if y is not None else self.y_train
        if target is None:
            raise RuntimeError("No hay etiquetas disponibles para calcular proporciones.")
        return target.value_counts(normalize=True).to_dict()