"""
Entrenamiento y validación cruzada de los modelos de clasificación binaria.
"""
import pandas as pd

from src.config import (
    MODEL_CONFIGS, MODEL_TAGS,
    N_FOLDS,
    get_logger
)
from .data_loader import DataLoader
from .evaluator import Evaluator
from .model_factory import build_and_fit, compute_class_weights
from .preprocessor import Preprocessor

logger = get_logger(__name__)


class Trainer:
    """
    Orquesta el entrenamiento con validación cruzada estratificada para
    los modelos indicados en `model_params` (o todos los de
    config.MODEL_CONFIGS si no se especifica ninguno). Agregar un modelo
    nuevo NO requiere tocar esta clase: basta con registrarlo en
    MODEL_CONFIGS.
    """

    def __init__(
        self,
        loader: DataLoader,
        dataset_tag: str = "full",
        model_params: dict[str, dict] | None = None,
    ) -> None:
        self.loader = loader
        self.dataset_tag = dataset_tag

        if model_params is not None:
            unknown = set(model_params) - set(MODEL_CONFIGS)
            if unknown:
                logger.error(f"Modelos no registrados en MODEL_CONFIGS: {sorted(unknown)}")
                raise ValueError(
                    f"Modelos desconocidos: {sorted(unknown)}. Disponibles: {list(MODEL_CONFIGS.keys())}"
                )
            self.model_tags: tuple[str, ...] = tuple(model_params.keys())
            self.model_params: dict[str, dict] = {tag: params.copy() for tag, params in model_params.items()}
        else:
            self.model_tags = MODEL_TAGS
            self.model_params = {tag: cfg["base_params"].copy() for tag, cfg in MODEL_CONFIGS.items()}

        self.fold_results = []
        self.final_models = {}
        logger.debug(f"Trainer inicializado para dataset: {dataset_tag} | Modelos a entrenar: {self.model_tags}")

    def run_cv(self) -> "Trainer":
        """Ejecuta la validación cruzada estratificada sobre X_train/y_train, solo para self.model_tags."""
        if self.loader.X_train is None or self.loader.y_train is None:
            logger.error("Intento de CV sin datos de entrenamiento.")
            raise RuntimeError("X_train / y_train no disponibles. Ejecuta split() en DataLoader.")

        if not self.model_tags:
            logger.warning("Trainer no tiene modelos asignados (model_tags vacío); run_cv() no hará nada.")
            return self

        folds = self.loader.get_cv_folds()
        logger.info(
            f"Iniciando Validación Cruzada ({N_FOLDS} folds) — dataset: {self.dataset_tag} "
            f"| Modelos: {self.model_tags}"
        )

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

                # 4. Entrenamiento SOLO de los modelos solicitados
                logger.debug(f"Fold {fold_idx + 1}: Entrenando modelos {self.model_tags}")
                models = {
                    tag: self._train_model(tag, X_fold_train_scaled, y_fold_train, weights)
                    for tag in self.model_tags
                }

                # 5. Evaluación local
                evaluator = Evaluator()
                metrics = {
                    tag: evaluator.evaluate(model, X_fold_val_scaled, y_fold_val)
                    for tag, model in models.items()
                }

                # 6. Almacenamiento
                self.fold_results.append({
                    "fold": fold_idx,
                    "dataset_tag": self.dataset_tag,
                    "weights": weights,
                    "models": models,
                    "preprocessor": prep,
                    "metrics": metrics,
                    "X_val": X_fold_val_scaled,
                    "y_val": y_fold_val,
                })
            except Exception as e:
                logger.error(f"Error crítico en el Fold {fold_idx + 1}: {e}")
                raise

        logger.info("Validación cruzada completada exitosamente.")
        return self

    def train_final(self) -> "Trainer":
        """Reentrena SOLO los modelos en self.model_tags sobre el 100% de X_train (sin folds)."""
        if self.loader.X_train is None or self.loader.y_train is None:
            logger.error("Intento de entrenamiento final sin datos.")
            raise RuntimeError("X_train / y_train no disponibles. Ejecuta split() en DataLoader.")

        if not self.model_tags:
            logger.warning("Trainer no tiene modelos asignados (model_tags vacío); train_final() no hará nada.")
            self.final_models = {}
            return self

        logger.info(f"Entrenando modelos finales sobre todo X_train ({self.dataset_tag}) | Modelos: {self.model_tags}")

        try:
            prep = Preprocessor(fold_id="final", dataset_tag=self.dataset_tag)
            X_scaled = prep.fit_transform(self.loader.X_train)
            prep.save()

            weights = self._compute_weights(self.loader.y_train)

            models = {
                tag: self._train_model(tag, X_scaled, self.loader.y_train, weights)
                for tag in self.model_tags
            }

            self.final_models = {
                **models,
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
        """Calcula pesos de clase balanceados (delegado a model_factory, fuente única)."""
        w_dict = compute_class_weights(y)
        logger.debug(f"Pesos de clase calculados: {w_dict}")
        return w_dict

    def _train_model(self, tag: str, X: pd.DataFrame, y: pd.Series, class_weights: dict):
        """
        Entrena el modelo `tag` delegando en model_factory.build_and_fit,
        que aplica el mecanismo de balanceo declarado en MODEL_CONFIGS
        (scale_pos_weight o sample_weight). Esto garantiza que Trainer y
        HyperparameterOptimizer entrenen bajo el mismo mecanismo.
        """
        return build_and_fit(tag, self.model_params[tag], X, y, class_weights=class_weights)