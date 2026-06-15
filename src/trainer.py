"""
Entrenamiento y validación cruzada de los modelos de clasificación binaria.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.config import (
    DT_BASE_PARAMS, XGBOOST_BASE_PARAMS,
    N_FOLDS,
    get_logger
)
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.preprocessor import Preprocessor

logger = get_logger(__name__)


class Trainer:
    """
    Orquesta el entrenamiento con validación cruzada estratificada para
    un árbol de decisión y un XGBoost.
    """

    def __init__(
        self,
        loader: DataLoader,
        dataset_tag: str = "full",
        dt_params: dict | None = None,
        xgb_params: dict | None = None,
    ) -> None:
        self.loader = loader
        self.dataset_tag = dataset_tag
        self.dt_params = dt_params or DT_BASE_PARAMS.copy()
        self.xgb_params = xgb_params or XGBOOST_BASE_PARAMS.copy()

        self.fold_results = []
        self.final_models = {}
        logger.debug(f"Trainer inicializado para dataset: {dataset_tag}")

    def run_cv(self) -> "Trainer":
        """Ejecuta la validación cruzada estratificada sobre X_train/y_train."""
        if self.loader.X_train is None or self.loader.y_train is None:
            logger.error("Intento de CV sin datos de entrenamiento.")
            raise RuntimeError("X_train / y_train no disponibles. Ejecuta split() en DataLoader.")

        folds = self.loader.get_cv_folds()
        logger.info(f"Iniciando Validación Cruzada ({N_FOLDS} folds) — dataset: {self.dataset_tag}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"[CV] Procesando Fold {fold_idx + 1}/{N_FOLDS}...")

            try:
                # 1. Extracción de datos del fold
                X_fold_train = self.loader.X_train.iloc[train_idx]
                y_fold_train = self.loader.y_train.iloc[train_idx]
                X_fold_val = self.loader.X_train.iloc[val_idx]
                y_fold_val = self.loader.y_train.iloc[val_idx]

                # 2. Preprocesamiento aislado
                prep = Preprocessor(fold_id=fold_idx, dataset_tag=self.dataset_tag)
                X_fold_train_scaled = prep.fit_transform(X_fold_train)
                X_fold_val_scaled = prep.transform(X_fold_val)
                prep.save()

                # 3. Pesos de clase para mitigar desbalance
                weights = self._compute_weights(y_fold_train)
                sample_w_tr = y_fold_train.map(weights)

                # 4. Entrenamiento de modelos
                logger.debug(f"Fold {fold_idx + 1}: Entrenando modelos (DT & XGBoost)")
                dt = self._train_dt(X_fold_train_scaled, y_fold_train, sample_w_tr)
                xgb = self._train_xgb(X_fold_train_scaled, y_fold_train, weights)

                # 5. Evaluación local
                evaluator = Evaluator()
                dt_metrics = evaluator.evaluate(dt, X_fold_val_scaled, y_fold_val)
                xgb_metrics = evaluator.evaluate(xgb, X_fold_val_scaled, y_fold_val)

                # 6. Almacenamiento
                self.fold_results.append({
                    "fold": fold_idx,
                    "dataset_tag": self.dataset_tag,
                    "weights": weights,
                    "models": {"dt": dt, "xgb": xgb},
                    "preprocessor": prep,
                    "metrics": {"dt": dt_metrics, "xgb": xgb_metrics},
                    "X_val": X_fold_val_scaled,
                    "y_val": y_fold_val,
                })
            except Exception as e:
                logger.error(f"Error crítico en el Fold {fold_idx + 1}: {e}")
                raise

        logger.info("Validación cruzada completada exitosamente.")
        return self

    def train_final(self) -> "Trainer":
        """Reentrena ambos modelos sobre el 100% de X_train (sin folds)."""
        if self.loader.X_train is None or self.loader.y_train is None:
            logger.error("Intento de entrenamiento final sin datos.")
            raise RuntimeError("X_train / y_train no disponibles. Ejecuta split() en DataLoader.")

        logger.info(f"Entrenando modelos finales sobre todo X_train ({self.dataset_tag})")
        
        try:
            prep = Preprocessor(fold_id="final", dataset_tag=self.dataset_tag)
            X_scaled = prep.fit_transform(self.loader.X_train)
            prep.save()

            weights = self._compute_weights(self.loader.y_train)
            sample_w = self.loader.y_train.map(weights)

            dt = self._train_dt(X_scaled, self.loader.y_train, sample_w)
            xgb = self._train_xgb(X_scaled, self.loader.y_train, weights)

            self.final_models = {
                "dt": dt,
                "xgb": xgb,
                "preprocessor": prep,
                "weights": weights,
            }
            logger.info("Entrenamiento final completado.")
        except Exception as e:
            logger.error(f"Fallo durante el entrenamiento final: {e}")
            raise

        return self

    def get_cv_summary(self) -> pd.DataFrame:
        """Agrega las métricas de todos los folds en un DataFrame resumido."""
        if not self.fold_results:
            logger.error("Intento de extraer resumen CV sin resultados.")
            raise RuntimeError("Ejecuta run_cv() antes de get_cv_summary().")

        rows = []
        for r in self.fold_results:
            row = {"fold": r["fold"], "dataset_tag": r["dataset_tag"]}
            for model_name, metrics in r["metrics"].items():
                for metric_name, value in metrics.items():
                    row[f"{model_name}_{metric_name}"] = value
            rows.append(row)

        df_summary = pd.DataFrame(rows)
        logger.debug(f"Resumen CV generado: {df_summary.shape} dimensiones.")
        return df_summary

    @staticmethod
    def _compute_weights(y: pd.Series) -> dict:
        """Calcula pesos de clase balanceados."""
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        w_dict = dict(zip(classes, weights))
        logger.debug(f"Pesos de clase calculados: {w_dict}")
        return w_dict

    def _train_dt(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series) -> DecisionTreeClassifier:
        """Entrena Decision Tree con pesos de instancia."""
        dt = DecisionTreeClassifier(**self.dt_params)
        dt.fit(X, y, sample_weight=sample_weight)
        return dt

    def _train_xgb(self, X: pd.DataFrame, y: pd.Series, class_weights: dict) -> XGBClassifier:
        """Entrena XGBoost usando scale_pos_weight estático para desbalance."""
        # scale_pos_weight = (peso_clase_0 / peso_clase_1)
        scale_pos_weight = class_weights[0] / class_weights[1]
        
        params = {**self.xgb_params, "scale_pos_weight": scale_pos_weight}
        xgb = XGBClassifier(**params)
        xgb.fit(X, y)
        return xgb