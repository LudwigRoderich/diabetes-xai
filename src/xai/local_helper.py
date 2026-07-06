"""
Módulo auxiliar para métodos de explicabilidad local (XAI) y análisis de estabilidad.
Proporciona herramientas agnósticas para extraer anclas de grupos previamente formados,
comparar explicaciones mediante Spearman y calcular la constante local de Lipschitz.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import Tuple, Any

from src.config import get_logger, FIGURES_XAI_DIR

logger = get_logger(__name__)


class LocalStabilityAnalyzer:
    """
    Clase base para gestionar explicaciones locales (LIME, SHAP) y medir su 
    robustez frente a perturbaciones en el espacio de características.
    """
    def __init__(self, model: Any, model_name: str, dataset_tag: str) -> None:
        self.model = model
        self.model_name = model_name
        self.dataset_tag = dataset_tag
        logger.debug(f"LocalStabilityAnalyzer inicializado para el modelo '{model_name}' ({dataset_tag}).")

    def load_neighborhood_and_anchor(self, filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lee un archivo CSV con un grupo de observaciones (vecindario) e identifica
        al ancla mediante el criterio de distancia de Gower igual a cero.
        """
        if not filepath.exists():
            logger.error(f"Archivo de vecindario no encontrado en la ruta: {filepath}")
            raise FileNotFoundError(f"No se encontró el grupo: {filepath}")

        df = pd.read_csv(filepath)
        
        if "gower_dist" not in df.columns:
            logger.error("El archivo proporcionado no contiene la métrica 'gower_dist'.")
            raise ValueError("El formato del archivo de vecindario es inválido.")

        # Uso de isclose en lugar de '==' estricto para evitar errores por precisión de coma flotante
        is_anchor = np.isclose(df["gower_dist"], 0.0, atol=1e-8)
        
        anchor_df = df[is_anchor].copy()
        neighbors_df = df[~is_anchor].copy()

        if anchor_df.empty:
            logger.warning("No se encontró ninguna observación con gower_dist == 0 en el grupo.")
        elif len(anchor_df) > 1:
            logger.warning(f"Se encontraron {len(anchor_df)} anclas (registros duplicados/idénticos).")
        else:
            logger.info(f"Grupo cargado con éxito: 1 ancla identificada y {len(neighbors_df)} vecinos.")

        return anchor_df, neighbors_df

    def plot_local_explanation(self, explanation: pd.Series, title: str, filename: str) -> Path:
        """
        Grafica un vector de explicación local (importancia de características)
        mediante un gráfico de barras horizontales divergente.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Ordenar características por impacto absoluto para visualizar las más influyentes arriba
        exp_sorted = explanation.reindex(explanation.abs().sort_values(ascending=True).index)

        # Colores divergentes: Azul (impacto positivo/hacia clase 1), Naranja (impacto negativo/hacia clase 0)
        colors = ["#DD8452" if val < 0 else "#4C72B0" for val in exp_sorted]
        
        ax.barh(exp_sorted.index, exp_sorted.values, color=colors, edgecolor="black", alpha=0.8) #type: ignore
        ax.set_title(title)
        ax.set_xlabel("Impacto local en la predicción (Log-Odds o Probabilidad)")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="-")
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        FIGURES_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_XAI_DIR / filename
        
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info(f"Gráfico local exportado exitosamente a: {out_path.name}")
        except Exception as e:
            logger.error(f"Fallo al guardar la figura {filename}: {e}")
        finally:
            plt.close(fig)

        return out_path

    def compute_spearman(self, exp1: np.ndarray, exp2: np.ndarray) -> float:
        """
        Calcula la correlación de rango de Spearman rho (p) entre dos 
        vectores de explicación, capturando la similitud en el ordenamiento.
        """
        # Suprimimos warnings en caso de vectores constantes (que generarían NaN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(exp1, exp2)

        # Si un vector no tiene varianza (ej. LIME devolvió ceros en todo), Spearman es indefinido
        if np.isnan(rho): #type: ignore
            logger.debug("Correlación de Spearman devolvió NaN. Asignando 0.0 por defecto.")
            return 0.0
            
        return float(rho) #type: ignore

    def compute_lipschitz(self, exp_anchor: np.ndarray, exp_neighbor: np.ndarray, gower_dist: float) -> float:
        """
        Calcula la constante de estabilidad local de Lipschitz para una observación
        y su vecino, basada en la correlación de Spearman y la distancia de Gower.
        """
        if gower_dist <= 1e-8:
            logger.warning("Distancia de Gower es $\approx 0$. Evitando división por cero.")
            return 0.0

        rho = self.compute_spearman(exp_anchor, exp_neighbor)
        
        # D_exp(xi, xj) = (1 - p(xi, xj)) / 2
        d_exp = (1.0 - rho) / 2.0
        
        # Lipschitz = D_exp / D_feat
        lipschitz_constant = d_exp / gower_dist

        return float(lipschitz_constant)
    
    @staticmethod
    def plot_stability_scatter(stability_csv: Path) -> Path:
        """
        Genera un panel dual de dispersión para analizar visualmente la degradación 
        de la explicación y el comportamiento de la inestabilidad en el vecindario.
        No depende de ningún modelo instanciado.
        """
        if not stability_csv.exists():
            logger.error(f"No se encontró el archivo de estabilidad: {stability_csv}")
            raise FileNotFoundError(f"Archivo no encontrado: {stability_csv}")

        df = pd.read_csv(stability_csv)
        
        # Validar contenido mínimo
        required_cols = ["gower_dist", "spearman_rho", "lipschitz_constant"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"El archivo carece de la columna métrica requerida: {col}")
                raise ValueError(f"Archivo inválido. Columna faltante: {col}")

        # Crear panel dual (1 fila, 2 columnas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Panel 1: Distancia de Gower vs Similitud de Explicación (Spearman)
        # El color representa la inestabilidad (Lipschitz)
        sc1 = ax1.scatter(
            df["gower_dist"], df["spearman_rho"], 
            c=df["lipschitz_constant"], cmap="viridis", 
            alpha=0.7, edgecolor="black", s=50
        )
        ax1.set_xlabel("Distancia de Gower (Similitud de Pacientes)")
        ax1.set_ylabel("Correlación de Spearman (Similitud de Interpretación)")
        ax1.set_title("Degradación de la Explicación vs Perturbación")
        ax1.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.6) # Línea donde la explicación se invierte
        ax1.grid(True, linestyle="--", alpha=0.4)
        cbar = plt.colorbar(sc1, ax=ax1)
        cbar.set_label("Constante de Lipschitz (Inestabilidad)")

        # Panel 2: Distancia de Gower vs Constante de Lipschitz directamente
        ax2.scatter(
            df["gower_dist"], df["lipschitz_constant"], 
            color="#DD8452", alpha=0.7, edgecolor="black", s=50
        )
        ax2.set_xlabel("Distancia de Gower (Similitud de Pacientes)")
        ax2.set_ylabel("Constante de Lipschitz")
        ax2.set_title("Magnitud de la Inestabilidad (Lipschitz)")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(f"Análisis de Estabilidad Espacial - {stability_csv.stem}", fontsize=14, y=1.02)
        plt.tight_layout()

        FIGURES_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"stability_scatter_{stability_csv.stem}.png"
        out_path = FIGURES_XAI_DIR / out_filename
        
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Panel de dispersión de estabilidad generado en: {out_path.name}")
        return out_path