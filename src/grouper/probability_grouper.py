import pandas as pd
import numpy as np
import gower
from src.config import get_logger

logger = get_logger(__name__)

class ProbabilityGrouper:
    def __init__(self, df: pd.DataFrame, probas: np.ndarray) -> None:
        self.df = df.copy()
        self.df["proba"] = probas
        self.df = self.df.sort_values(by="proba").reset_index(drop=True)
        logger.debug("ProbabilityGrouper inicializado y datos ordenados por probabilidad.")

    def get_subgroup(self, min_prob: float, max_prob: float) -> pd.DataFrame:
        subgroup = self.df[(self.df["proba"] >= min_prob) & (self.df["proba"] <= max_prob)].copy()
        logger.info(f"Subgrupo extraído [{min_prob}, {max_prob}]: {len(subgroup)} observaciones.")
        return subgroup

    def get_anchor(self, subgroup: pd.DataFrame, target_prob: float, min_prob: float, max_prob: float) -> pd.DataFrame:
        """Extrae la observación más cercana a la probabilidad objetivo, validando que pertenezca al intervalo."""
        if target_prob < min_prob or target_prob > max_prob:
            logger.error(f"Probabilidad objetivo ({target_prob}) fuera del intervalo permitido [{min_prob}, {max_prob}].")
            raise ValueError(f"El ancla debe pertenecer al intervalo definido.")
            
        if subgroup.empty:
            logger.error("Intento de extraer ancla de un subgrupo vacío.")
            raise ValueError("El subgrupo proporcionado está vacío.")
            
        idx = (subgroup["proba"] - target_prob).abs().idxmin()
        anchor = subgroup.loc[[idx]].copy()
        logger.info(f"Ancla seleccionada con probabilidad real: {anchor['proba'].values[0]:.4f}")
        return anchor

    def get_neighborhood(self, subgroup: pd.DataFrame, anchor: pd.DataFrame, radius: float, feature_cols: list) -> pd.DataFrame:
        if subgroup.empty or anchor.empty:
            logger.error("Subgrupo o ancla vacíos al calcular vecindad.")
            raise ValueError("Subgrupo y ancla deben contener datos.")

        anchor_calc = anchor[feature_cols].astype(np.float64)
        subgroup_calc = subgroup[feature_cols].astype(np.float64)

        dist_matrix = gower.gower_matrix(anchor_calc, subgroup_calc)
        
        result = subgroup.copy()
        result["gower_dist"] = dist_matrix[0]
        
        neighborhood = result[result["gower_dist"] <= radius].copy()
        logger.info(f"Vecindad calculada (radio <= {radius}): {len(neighborhood)} observaciones encontradas.")
        return neighborhood