"""
Permutation Feature Importance (PFI) global para clasificación binaria.

Cuantifica la relevancia de cada característica midiendo la caída en PR-AUC
cuando sus valores son permutados aleatoriamente sobre el conjunto de validación.
Una caída mayor indica mayor dependencia del modelo respecto a esa característica.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import(
    PFI_N_REPEATS, 
    PFI_SEED, 
    PFI_MAX_SAMPLES, 
    PFI_SCORING, 
    PFI_NJOBS, 
    get_logger
)

logger = get_logger(__name__)


class PermutationImportanceAnalyzer:
    """
    Encapsula el análisis PFI global para un clasificador binario ya entrenado.

    El modelo debe exponer predict_proba() y los datos de entrada deben estar
    preprocesados con xai_mode=False desde Preprocessor, que es el mismo espacio
    de características con el que el modelo fue entrenado.
    """

    def __init__(
        self,
        model,
        model_name: str,
        dataset_tag: str,
        n_repeats: int = PFI_N_REPEATS,
        random_state: int = PFI_SEED,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.dataset_tag = dataset_tag
        self.n_repeats = n_repeats
        self.random_state = random_state

        self._raw_result = None
        self._importance_df: pd.DataFrame | None = None

        logger.debug(
            f"PermutationImportanceAnalyzer inicializado — "
            f"modelo: {model_name.upper()} | dataset: {dataset_tag} | n_repeats: {n_repeats}"
        )

    def compute(self, X: pd.DataFrame, y: pd.Series) -> "PermutationImportanceAnalyzer":
        """
        Ejecuta el cómputo PFI sobre (X, y) y almacena los resultados internamente.

        X debe ser X_val_scaled obtenido con prep.transform(X_val_raw, xai_mode=False).
        Usar xai_mode=True produciría un espacio de características diferente al que
        el modelo vio durante el entrenamiento, invalidando las predicciones.

        Retorna self para encadenamiento opcional con get_importance_df().
        """
        logger.info(
            f"[PFI] Ejecutando análisis — modelo: {self.model_name.upper()} | "
            f"dataset: {self.dataset_tag} | instancias: {len(X)} | features: {X.shape[1]}"
        )

        try:
            self._raw_result = permutation_importance(
                estimator=self.model,
                X=X,
                y=y,
                scoring=PFI_SCORING,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=PFI_NJOBS,
                max_samples=PFI_MAX_SAMPLES,
            )
        except Exception as e:
            logger.error(f"[PFI] Fallo durante permutation_importance: {e}")
            raise

        df = ( 
            pd.DataFrame({
                "feature": X.columns.tolist(),
                "importance_mean": self._raw_result.importances_mean, #type: ignore
                "importance_std": self._raw_result.importances_std,   #type: ignore
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        self._importance_df = df

        top = df.iloc[0]
        logger.info(
            f"[PFI] Completado — feature más importante: '{top['feature']}' "
            f"(mean={top['importance_mean']:.4f}, std={top['importance_std']:.4f})"
        )

        return self

    def get_importance_df(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame completo de importancias ordenado por rank.
        Columnas: feature, importance_mean, importance_std, rank.

        importance_mean negativa indica que permutar esa característica mejora el PR-AUC,
        lo que sugiere colinealidad o que el modelo no la usa de forma discriminativa.
        """
        self._check_computed()
        if self._importance_df is None:
            logger.error(f"Intento de acceder a importance_df sin resultados computados — modelo: {self.model_name.upper()}")
            raise RuntimeError("Ejecuta compute() antes de acceder a importance_df.")
        return self._importance_df.copy()

    def get_ranking(self) -> pd.Series:
        """
        Devuelve un mapeo feature → rank (1 = más importante).

        Entrada directa para la correlación de Spearman contra el ranking intrínseco
        del árbol de decisión (feature_importances_) y contra LIME y SHAP.
        """
        self._check_computed()
        if self._importance_df is None:
            logger.error(f"Intento de acceder a ranking sin resultados computados — modelo: {self.model_name.upper()}")
            raise RuntimeError("Ejecuta compute() antes de acceder al ranking.")
        return self._importance_df.set_index("feature")["rank"].copy()

    def get_raw_importances(self) -> np.ndarray:
        """
        Devuelve la matriz de caídas individuales por repetición.
        Forma: (n_features, n_repeats).

        Permite analizar la varianza interna del PFI entre repeticiones,
        útil para diagnosticar estabilidad del análisis antes del módulo formal de estabilidad.
        """
        self._check_computed()
        return self._raw_result.importances.copy() # type: ignore

    def _check_computed(self) -> None:
        if self._importance_df is None:
            logger.error(
                f"Acceso a resultados PFI sin ejecutar compute() — "
                f"modelo: {self.model_name.upper()}"
            )
            raise RuntimeError("Ejecuta compute() antes de acceder a los resultados del PFI.")