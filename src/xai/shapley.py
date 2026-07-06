"""
Módulo especializado para la explicabilidad local utilizando SHAP (SHapley Additive exPlanations).
Aprovecha TreeExplainer para modelos basados en árboles (DT, XGBoost) y calcula
la estabilidad local frente al vecindario.
"""
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Any

from src.config import get_logger, CONTINUOUS_COLS, RESULTS_XAI_DIR
from src.xai.local_helper import LocalStabilityAnalyzer
from src.preprocessor import Preprocessor

logger = get_logger(__name__)

class ShapLocalAnalyzer(LocalStabilityAnalyzer):
    """
    Extiende LocalStabilityAnalyzer para calcular y evaluar explicaciones locales
    usando los valores exactos de Shapley a través de shap.TreeExplainer.
    """
    def __init__(self, model: Any, model_name: str, dataset_tag: str) -> None:
        super().__init__(model, model_name, dataset_tag)
        
    def _get_shap_values_class1(self, explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
        """
        Extrae los valores SHAP correspondientes a la clase positiva (1).
        Maneja las diferencias estructurales en las salidas de sklearn (lista) 
        y XGBoost (matriz directa).
        """
        shap_vals = explainer.shap_values(X)
        
        # Scikit-Learn DecisionTree retorna una lista de matrices [clase_0, clase_1]
        if isinstance(shap_vals, list):
            return shap_vals[1]
        
        # XGBoost binario retorna una única matriz de log-odds para la clase 1
        return shap_vals

    def analyze_stability(self, neighborhood_path: Path, feature_cols: List[str], preprocessor: Optional[Preprocessor] = None) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de XAI local:
        1. Carga el vecindario y ubica el ancla.
        2. Escala las variables continuas si se provee un preprocesador.
        3. Calcula los vectores SHAP para ancla y vecinos.
        4. Grafica el impacto local del ancla.
        5. Computa la constante de Lipschitz para evaluar la estabilidad de cada vecino.
        """
        logger.info(f"Iniciando análisis de estabilidad SHAP local para '{self.model_name}'...")
        
        # 1. Cargar datos usando la clase padre
        anchor_df, neighbors_df = self.load_neighborhood_and_anchor(neighborhood_path)
        if anchor_df.empty:
            logger.error("Se aborta SHAP local: No se identificó ningún ancla válida en el archivo.")
            return pd.DataFrame()

        # 2. Extraer características
        anchor_X = anchor_df[feature_cols].copy()
        neighbors_X = neighbors_df[feature_cols].copy() if not neighbors_df.empty else pd.DataFrame(columns=feature_cols)

        # 3. Aplicar escalado si es requerido (el modelo fue entrenado con datos escalados)
        if preprocessor is not None:
            anchor_X[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(anchor_X[CONTINUOUS_COLS])
            if not neighbors_X.empty:
                neighbors_X[CONTINUOUS_COLS] = preprocessor._scaler_model.transform(neighbors_X[CONTINUOUS_COLS])
            logger.debug("Variables continuas escaladas previo a calcular SHAP.")

        # 4. Inicializar Explainer
        try:
            # TreeExplainer es ideal para DT y XGBoost
            explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.error(f"Fallo al inicializar shap.TreeExplainer: {e}")
            raise

        # 5. Calcular SHAP para el ancla y graficar
        anchor_shap_matrix = self._get_shap_values_class1(explainer, anchor_X)
        anchor_shap_vector = anchor_shap_matrix[0] # Al ser un ancla, es el primer y único elemento
        
        anchor_series = pd.Series(anchor_shap_vector, index=feature_cols)
        file = f"{neighborhood_path.stem}-SHAP_Local_Impact ({self.model_name.upper()})"
        
        self.plot_local_explanation(
            explanation=anchor_series,
            title=f"Impacto Local SHAP - Ancla ({self.model_name.upper()})",
            filename=f"{file}.png"
        )

        # 6. Analizar estabilidad contra los vecinos
        results = []
        if not neighbors_X.empty:
            neighbors_shap_matrix = self._get_shap_values_class1(explainer, neighbors_X)
            
            for idx, (_, neighbor_row) in enumerate(neighbors_df.iterrows()):
                n_shap_vector = neighbors_shap_matrix[idx]
                gower_dist = neighbor_row["gower_dist"]
                
                # Usar métodos base de LocalStabilityAnalyzer
                rho = self.compute_spearman(anchor_shap_vector, n_shap_vector)
                lipschitz = self.compute_lipschitz(anchor_shap_vector, n_shap_vector, gower_dist)
                
                results.append({
                    "neighbor_index": idx,
                    "gower_dist": gower_dist,
                    "spearman_rho": rho,
                    "lipschitz_constant": lipschitz
                })
        else:
            logger.warning("No hay vecinos disponibles en el radio. No se evaluará estabilidad.")

        # 7. Guardar resultados
        results_df = pd.DataFrame(results)
        RESULTS_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_XAI_DIR / f"shap_stability_{neighborhood_path.stem}.csv"
        
        if not results_df.empty:
            results_df.to_csv(out_path, index=False)
            logger.info(f"Reporte de estabilidad SHAP guardado en: {out_path.name}")
            logger.info(f"Lipschitz promedio del vecindario: {results_df['lipschitz_constant'].mean():.4f}")

        return results_df