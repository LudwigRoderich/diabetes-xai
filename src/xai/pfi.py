"""
Permutation Feature Importance (PFI) global para clasificación binaria.

Cuantifica la relevancia de cada característica midiendo la caída en PR-AUC
cuando sus valores son permutados aleatoriamente sobre el conjunto de validación.
Una caída mayor indica mayor dependencia del modelo respecto a esa característica.
"""
import numpy as np
import pandas as pd
from datetime import datetime
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

    def compute(self, X: pd.DataFrame, y: pd.Series) -> "PermutationImportanceAnalyzer":
        
        logger.info("\n\n" + "=" * 80)
        logger.info("=== NUEVA ITERACIÓN PFI ===")
        logger.info(f"Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Modelo: {self.model_name.upper()} | Dataset Tag: {self.dataset_tag}")
        logger.info(f"Parámetros: n_repeats={self.n_repeats}, scoring={PFI_SCORING}, max_samples={PFI_MAX_SAMPLES}, seed={self.random_state}")
        logger.info("=" * 80 + "\n")

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
        
        logger.info("\n" + "-" * 50)
        logger.info("RESULTADOS FINALES DE LA ITERACIÓN PFI:")
        logger.info(f"Feature más importante detectada: '{top['feature']}' (mean={top['importance_mean']:.4f}, std={top['importance_std']:.4f})")
        logger.info("-" * 50 + "\n\n")

        return self

    def get_importance_df(self) -> pd.DataFrame:
        self._check_computed()
        if self._importance_df is None:
            raise RuntimeError("Ejecuta compute() antes de acceder a importance_df.")
        return self._importance_df.copy()

    def get_ranking(self) -> pd.Series:
        self._check_computed()
        if self._importance_df is None:
            raise RuntimeError("Ejecuta compute() antes de acceder al ranking.")
        return self._importance_df.set_index("feature")["rank"].copy()

    def get_raw_importances(self) -> np.ndarray:
        self._check_computed()
        return self._raw_result.importances.copy() # type: ignore

    def _check_computed(self) -> None:
        if self._importance_df is None:
            logger.error(
                f"Acceso a resultados PFI sin ejecutar compute() — "
                f"modelo: {self.model_name.upper()}"
            )
            raise RuntimeError("Ejecuta compute() antes de acceder a los resultados del PFI.")