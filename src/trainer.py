"""
Entrenamiento y validación cruzada de los modelos de clasificación binaria.

Flujo por fold:
  1. Ajustar y aplicar el Preprocessor sobre el subconjunto de entrenamiento.
  2. Calcular pesos de clase a partir de y_train del fold.
  3. Entrenar DecisionTreeClassifier y XGBClassifier.
  4. Evaluar sobre el subconjunto de validación del fold.
  5. Devolver resultados estructurados para su posterior guardado.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import (
    CONTINUOUS_COLS, ORDINAL_COLS,
    DT_BASE_PARAMS, XGBOOST_BASE_PARAMS,
    N_FOLDS, GLOBAL_SEED,
    TARGET_COL,
)
from data_loader import DataLoader
from evaluator import Evaluator
from preprocessor import Preprocessor


class Trainer:
    """
    Orquesta el entrenamiento con validación cruzada estratificada para
    un árbol de decisión y un XGBoost, sobre el conjunto de entrenamiento
    provisto por DataLoader.

    Parameters
    ----------
    loader : DataLoader
        Instancia ya cargada y particionada (load() y split() ejecutados).
    dataset_tag : str
        Etiqueta del dataset: 'full' o 'clean'. Se propaga al Preprocessor
        y a los resultados para identificar el origen de los datos.
    dt_params : dict, optional
        Parámetros para DecisionTreeClassifier. Si es None, usa DT_BASE_PARAMS.
    xgb_params : dict, optional
        Parámetros para XGBClassifier. Si es None, usa XGBOOST_BASE_PARAMS.

    Attributes
    ----------
    fold_results : list of dict
        Lista con los resultados de cada fold tras ejecutar run_cv().
    final_models : dict
        Modelos reentrenados sobre el 100 % de X_train al finalizar el CV.
    """

    def __init__(
        self,
        loader: DataLoader,
        dataset_tag: str = "full",
        dt_params: dict | None = None,
        xgb_params: dict | None = None,
    ) -> None:
        self.loader      = loader
        self.dataset_tag = dataset_tag
        self.dt_params   = dt_params  or DT_BASE_PARAMS.copy()
        self.xgb_params  = xgb_params or XGBOOST_BASE_PARAMS.copy()

        self.fold_results  = []
        self.final_models  = {}

    # ------------------------------------------------------------------
    def run_cv(self) -> "Trainer":
        """
        Ejecuta la validación cruzada estratificada de N_FOLDS folds
        sobre X_train / y_train del DataLoader.

        Por cada fold:
          - Ajusta un Preprocessor, lo guarda en disco.
          - Calcula pesos de clase.
          - Entrena DT y XGBoost.
          - Evalúa sobre el subconjunto de validación del fold.
          - Almacena el resultado en self.fold_results.

        Returns
        -------
        self
        """
        folds = self.loader.get_cv_folds()
        if self.loader.X_train is None or self.loader.y_train is None:
            raise RuntimeError("X_train / y_train no disponibles. Asegúrate de ejecutar split() en DataLoader.")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"[CV] Fold {fold_idx + 1}/{N_FOLDS} — dataset: {self.dataset_tag}")

            # ---- datos del fold ------------------------------------
            X_fold_train = self.loader.X_train.iloc[train_idx]
            y_fold_train = self.loader.y_train.iloc[train_idx]
            X_fold_val   = self.loader.X_train.iloc[val_idx]
            y_fold_val   = self.loader.y_train.iloc[val_idx]

            # ---- preprocesamiento ----------------------------------
            prep = Preprocessor(fold_id=fold_idx, dataset_tag=self.dataset_tag)
            X_fold_train_scaled = prep.fit_transform(X_fold_train)
            X_fold_val_scaled   = prep.transform(X_fold_val)
            prep.save()

            # ---- pesos de clase ------------------------------------
            weights      = self._compute_weights(y_fold_train)
            sample_w_tr  = y_fold_train.map(weights)

            # ---- entrenamiento -------------------------------------
            dt  = self._train_dt(X_fold_train_scaled, y_fold_train, sample_w_tr)
            xgb = self._train_xgb(X_fold_train_scaled, y_fold_train, weights)

            # ---- evaluación ----------------------------------------
            evaluator = Evaluator()
            dt_metrics  = evaluator.evaluate(dt,  X_fold_val_scaled, y_fold_val)
            xgb_metrics = evaluator.evaluate(xgb, X_fold_val_scaled, y_fold_val)

            self.fold_results.append({
                "fold"        : fold_idx,
                "dataset_tag" : self.dataset_tag,
                "weights"     : weights,
                "models"      : {"dt": dt, "xgb": xgb},
                "preprocessor": prep,
                "metrics"     : {"dt": dt_metrics, "xgb": xgb_metrics},
                "X_val"       : X_fold_val_scaled,  # Datos de validación del fold (escalados)
                "y_val"       : y_fold_val,         # Etiquetas de validación del fold
            })

        return self

    # ------------------------------------------------------------------
    def train_final(self) -> "Trainer":
        """
        Reentrena ambos modelos sobre el 100 % de X_train (sin folds)
        usando el Preprocessor ajustado sobre ese mismo conjunto completo.

        El modelo final es el que se usará para la fase XAI.
        El Preprocessor se guarda con fold_id='final'.

        Returns
        -------
        self
        """
        print(f"[FINAL] Entrenando sobre X_train completo — dataset: {self.dataset_tag}")
        if self.loader.X_train is None or self.loader.y_train is None:
            raise RuntimeError("X_train / y_train no disponibles. Asegúrate de ejecutar split() en DataLoader.")
        prep = Preprocessor(fold_id="final", dataset_tag=self.dataset_tag)
        X_scaled = prep.fit_transform(self.loader.X_train)
        prep.save()

        weights     = self._compute_weights(self.loader.y_train)
        sample_w    = self.loader.y_train.map(weights)

        dt  = self._train_dt(X_scaled, self.loader.y_train, sample_w)
        xgb = self._train_xgb(X_scaled, self.loader.y_train, weights)

        self.final_models = {
            "dt"          : dt,
            "xgb"         : xgb,
            "preprocessor": prep,
            "weights"     : weights,
        }
        return self

    # ------------------------------------------------------------------
    def get_cv_summary(self) -> pd.DataFrame:
        """
        Agrega las métricas de todos los folds en un DataFrame resumido.

        Returns
        -------
        pd.DataFrame
            Filas: folds. Columnas: modelo_métrica (ej. dt_prauc, xgb_recall).
        """
        if self.fold_results is None or len(self.fold_results) == 0:
            raise RuntimeError("Ejecuta run_cv() antes de get_cv_summary().")

        rows = []
        for r in self.fold_results:
            row = {"fold": r["fold"], "dataset_tag": r["dataset_tag"]}
            for model_name, metrics in r["metrics"].items():
                for metric_name, value in metrics.items():
                    row[f"{model_name}_{metric_name}"] = value
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weights(y: pd.Series) -> dict:
        """
        Calcula pesos de clase balanceados usando la fórmula estándar:
        w_i = n_samples / (n_classes * n_samples_i).

        Parameters
        ----------
        y : pd.Series
            Etiquetas del fold de entrenamiento.

        Returns
        -------
        dict : {clase: peso}
        """
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
        )
        return dict(zip(classes, weights))

    # ------------------------------------------------------------------
    def _train_dt(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series,
    ) -> DecisionTreeClassifier:
        """
        Instancia y entrena un DecisionTreeClassifier con los parámetros
        definidos en self.dt_params.

        Parameters
        ----------
        X : pd.DataFrame
            Features escaladas del fold de entrenamiento.
        y : pd.Series
            Etiquetas del fold de entrenamiento.
        sample_weight : pd.Series
            Peso por instancia derivado de los pesos de clase.

        Returns
        -------
        DecisionTreeClassifier entrenado.
        """
        dt = DecisionTreeClassifier(**self.dt_params)
        dt.fit(X, y, sample_weight=sample_weight)
        return dt

    # ------------------------------------------------------------------
    def _train_xgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_weights: dict,
    ) -> XGBClassifier:
        """
        Instancia y entrena un XGBClassifier. El parámetro scale_pos_weight
        se calcula como n_negativos / n_positivos a partir de class_weights.

        Parameters
        ----------
        X : pd.DataFrame
            Features escaladas del fold de entrenamiento.
        y : pd.Series
            Etiquetas del fold de entrenamiento.
        class_weights : dict
            Pesos de clase {0: w0, 1: w1} calculados por _compute_weights().

        Returns
        -------
        XGBClassifier entrenado.
        """
        # scale_pos_weight es el cociente de los pesos, no de los conteos,
        # porque _compute_weights() ya invirtió la proporción.
        scale_pos_weight = class_weights[0] / class_weights[1]

        params = {**self.xgb_params, "scale_pos_weight": scale_pos_weight}
        xgb = XGBClassifier(**params)
        xgb.fit(X, y)
        return xgb