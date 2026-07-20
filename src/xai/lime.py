"""
Módulo especializado para la explicabilidad local utilizando LIME
(Local Interpretable Model-Agnostic Explanations).

Implementa el procedimiento de Ribeiro et al. (2016) y ajusta un modelo surrogate
interpretable sobre las predicciones del modelo de caja negra en ese vecindario.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    get_logger,
    RESULTS_XAI_DIR,
    CONTINUOUS_COLS,
    ORDINAL_COLS,
    LIME_SEED,
    LIME_N_SAMPLES,
    LIME_KERNEL_WIDTH,
    LIME_NOISE_FACTOR,
    LIME_ORDINAL_PERTURB_PROB,
    LIME_BINARY_FLIP_PROB,
    LIME_SURROGATE_CHOICES,
    LOCAL_DISTANCE_COL,
)
from src.xai.local_helper import LocalStabilityAnalyzer
from src.binary_clasification.preprocessor import Preprocessor

logger = get_logger(__name__)


class LimeLocalAnalyzer(LocalStabilityAnalyzer):
    """
    Extiende LocalStabilityAnalyzer para producir explicaciones locales vía
    LIME, con el surrogate ("ridge" o "tree") fijado en la construcción.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        dataset_tag: str,
        surrogate: str = LIME_SURROGATE_CHOICES[0],
        seed: int = LIME_SEED,
    ) -> None:
        super().__init__(model, model_name, dataset_tag)

        if surrogate not in LIME_SURROGATE_CHOICES:
            logger.error(f"Surrogate '{surrogate}' inválido. Opciones válidas: {LIME_SURROGATE_CHOICES}")
            raise ValueError(f"Surrogate desconocido: {surrogate}")

        self.surrogate = surrogate
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _split_feature_types(feature_cols: List[str]) -> Dict[str, List[str]]:
        continuous = [c for c in feature_cols if c in CONTINUOUS_COLS]
        ordinal = [c for c in feature_cols if c in ORDINAL_COLS]
        binary = [c for c in feature_cols if c not in continuous and c not in ordinal]
        return {"continuous": continuous, "ordinal": ordinal, "binary": binary}

    def _compute_feature_stats(
        self, reference_df: pd.DataFrame, feature_types: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        cont = feature_types["continuous"]
        if cont:
            std = reference_df[cont].std(ddof=0)
            stats["continuous_std"] = std.replace(0, 1e-8)

        ordi = feature_types["ordinal"]
        if ordi:
            stats["ordinal_min"] = reference_df[ordi].min()
            stats["ordinal_max"] = reference_df[ordi].max()

        all_cols = feature_types["continuous"] + feature_types["ordinal"] + feature_types["binary"]
        ranges = reference_df[all_cols].max() - reference_df[all_cols].min()
        ranges = ranges.replace(0, 1.0)
        stats["feature_ranges"] = ranges

        return stats

    def _perturb_instance(
        self,
        anchor_raw: pd.Series,
        feature_types: Dict[str, List[str]],
        stats: Dict[str, Any],
        n_samples: int,
    ) -> pd.DataFrame:
        cont, ordi, bina = feature_types["continuous"], feature_types["ordinal"], feature_types["binary"]

        perturbed = pd.DataFrame(
            np.tile(anchor_raw.values.astype(float), (n_samples, 1)), #type: ignore
            columns=anchor_raw.index,
        )

        if cont:
            std = stats["continuous_std"][cont].values
            noise = self._rng.normal(loc=0.0, scale=1.0, size=(n_samples, len(cont)))
            perturbed[cont] = anchor_raw[cont].values + noise * (std * LIME_NOISE_FACTOR)

        if ordi:
            ord_min = stats["ordinal_min"][ordi].values
            ord_max = stats["ordinal_max"][ordi].values
            shift_mask = self._rng.random((n_samples, len(ordi))) < LIME_ORDINAL_PERTURB_PROB
            shift_dir = self._rng.choice([-1.0, 1.0], size=(n_samples, len(ordi)))
            shifted = perturbed[ordi].values + shift_mask * shift_dir
            perturbed[ordi] = np.clip(shifted, ord_min, ord_max)

        if bina:
            flip_mask = self._rng.random((n_samples, len(bina))) < LIME_BINARY_FLIP_PROB
            perturbed[bina] = np.where(flip_mask, 1.0 - perturbed[bina].values, perturbed[bina].values)

        return perturbed

    def _compute_kernel_weights(
        self,
        perturbed_raw: pd.DataFrame,
        anchor_raw: pd.Series,
        stats: Dict[str, Any],
        n_features: int,
    ) -> np.ndarray:
        ranges = stats["feature_ranges"][perturbed_raw.columns]
        normalized_diff = (perturbed_raw - anchor_raw) / ranges
        sq_dist = (normalized_diff ** 2).sum(axis=1).to_numpy()

        kernel_width = LIME_KERNEL_WIDTH if LIME_KERNEL_WIDTH is not None else 0.75 * np.sqrt(n_features)
        weights = np.sqrt(np.exp(-sq_dist / (kernel_width ** 2)))
        return weights

    def _fit_surrogate(
        self,
        X_surrogate: np.ndarray,
        y_target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[np.ndarray, Any]:
        if self.surrogate == "ridge":
            reg = Ridge(alpha=1.0, random_state=self.seed)
            reg.fit(X_surrogate, y_target, sample_weight=sample_weight)
            return reg.coef_, reg

        if self.surrogate == "tree":
            reg = DecisionTreeRegressor(max_depth=4, random_state=self.seed)
            reg.fit(X_surrogate, y_target, sample_weight=sample_weight)
            return reg.feature_importances_, reg

        raise ValueError(f"Surrogate no soportado: {self.surrogate}")

    @staticmethod
    def _weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        weighted_mean = np.average(y_true, weights=weights)
        ss_res = np.sum(weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)
        if ss_tot <= 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def explain_instance(
        self,
        instance_raw: pd.Series,
        feature_types: Dict[str, List[str]],
        stats: Dict[str, Any],
        preprocessor: Optional[Preprocessor],
        n_samples: int = LIME_N_SAMPLES,
    ) -> tuple[pd.Series, float]:
        feature_cols = instance_raw.index.tolist()
        n_features = len(feature_cols)

        perturbed_raw = self._perturb_instance(instance_raw, feature_types, stats, n_samples)
        weights = self._compute_kernel_weights(perturbed_raw, instance_raw, stats, n_features)

        perturbed_for_model = perturbed_raw.copy()
        if preprocessor is not None and feature_types["continuous"]:
            perturbed_for_model[feature_types["continuous"]] = preprocessor._scaler_model.transform(
                perturbed_raw[feature_types["continuous"]]
            )

        y_target = self.model.predict_proba(perturbed_for_model[feature_cols])[:, 1]
        X_surrogate = perturbed_for_model[feature_cols].to_numpy()
        coefs, fitted_surrogate = self._fit_surrogate(X_surrogate, y_target, weights)

        y_pred = fitted_surrogate.predict(X_surrogate)
        fidelity = self._weighted_r2(y_target, y_pred, weights)

        return pd.Series(coefs, index=feature_cols), fidelity

    def analyze_stability(
        self,
        neighborhood_path: Path,
        feature_cols: List[str],
        reference_df: pd.DataFrame,
        preprocessor: Optional[Preprocessor] = None,
        n_samples: int = LIME_N_SAMPLES,
    ) -> pd.DataFrame:
        
        logger.info("\n\n" + "=" * 80)
        logger.info("=== NUEVA ITERACIÓN LIME ===")
        logger.info(f"Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Modelo: {self.model_name.upper()} | Dataset Tag: {self.dataset_tag}")
        logger.info(f"Parámetros: surrogate='{self.surrogate}', seed={self.seed}, n_samples={n_samples}, kernel_width={LIME_KERNEL_WIDTH}")
        logger.info("=" * 80 + "\n")

        anchor_df, neighbors_df = self.load_neighborhood_and_anchor(neighborhood_path)
        if anchor_df.empty:
            logger.error("Se aborta LIME local: no se identificó ningún ancla válida en el archivo.")
            return pd.DataFrame()

        feature_types = self._split_feature_types(feature_cols)
        stats = self._compute_feature_stats(reference_df[feature_cols], feature_types)

        anchor_raw = anchor_df[feature_cols].iloc[0]
        anchor_explanation, anchor_fidelity = self.explain_instance(
            anchor_raw, feature_types, stats, preprocessor, n_samples
        )
        logger.info(f"[LIME] Ancla explicada. Fidelidad local (R²): {anchor_fidelity:.4f}.")

        file_stem = f"lime_local_impact_index_{anchor_df.index[0]}_{self.model_name}_{self.dataset_tag}_{self.surrogate}_{n_samples}samples_{neighborhood_path.stem}"
        self.plot_local_explanation(
            explanation=anchor_explanation,
            title=f"Impacto Local LIME [{self.surrogate[:1].upper()}{self.surrogate[1:]}] - Ancla",
            filename=f"{file_stem}.png",
        )

        results = []
        if not neighbors_df.empty:
            n_neighbors = len(neighbors_df)
            for idx, (_, neighbor_row) in enumerate(neighbors_df.iterrows()):
                neighbor_raw = neighbor_row[feature_cols]
                neighbor_explanation, neighbor_fidelity = self.explain_instance(
                    neighbor_raw, feature_types, stats, preprocessor, n_samples
                )

                # Extraer distancia dinámicamente configurada
                distance_val = neighbor_row[LOCAL_DISTANCE_COL]
                rho = self.compute_spearman(anchor_explanation.to_numpy(), neighbor_explanation.to_numpy())
                lipschitz = self.compute_lipschitz(
                    anchor_explanation.to_numpy(), neighbor_explanation.to_numpy(), distance_val
                )

                logger.info(
                    f"[LIME] Vecino {idx + 1}/{n_neighbors} procesado: distance={distance_val:.4f}, "
                    f"spearman_rho={rho:.4f}, lipschitz={lipschitz:.4f}, fidelity={neighbor_fidelity:.4f}."
                )

                results.append(
                    {
                        "neighbor_index": idx,
                        "distance": distance_val, # Guardado agnóstico como "distance"
                        "spearman_rho": rho,
                        "lipschitz_constant": lipschitz,
                        "anchor_fidelity": anchor_fidelity,
                        "neighbor_fidelity": neighbor_fidelity,
                    }
                )
        else:
            logger.warning("No hay vecinos disponibles en el radio. No se evaluará estabilidad.")

        results_df = pd.DataFrame(results)
        RESULTS_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_XAI_DIR / f"lime_stability_{self.surrogate}_{neighborhood_path.stem}.csv"

        if not results_df.empty:
            results_df.to_csv(out_path, index=False)
            
            logger.info("\n" + "-" * 50)
            logger.info("RESULTADOS FINALES DE LA ITERACIÓN LIME:")
            logger.info(f"Reporte de estabilidad guardado en: {out_path.name}")
            logger.info(f"Explicacion del ancla guardada en: {file_stem}.png")
            logger.info(f"Lipschitz promedio del vecindario: {results_df['lipschitz_constant'].mean():.4f}")
            logger.info(f"Fidelidad promedio de vecinos (R²): {results_df['neighbor_fidelity'].mean():.4f}")
            logger.info("-" * 50 + "\n\n")

        return results_df