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
from typing import Tuple, Any, Optional

from src.config import (
    get_logger, 
    FIGURES_XAI_DIR, 
    RESULTS_XAI_DIR,
    LOCAL_DISTANCE_COL, 
    LOCAL_ANCHOR_COL,
    STABILITY_SPEARMAN_THRESHOLD,
    STABILITY_LIPSCHITZ_THRESHOLD
)

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

    def load_neighborhood_and_anchor(self, filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lee un archivo CSV con un grupo de observaciones (vecindario) e identifica
        al ancla de forma agnóstica basándose en la configuración global.
        """
        if not filepath.exists():
            logger.error(f"Archivo de vecindario no encontrado en la ruta: {filepath}")
            raise FileNotFoundError(f"No se encontró el grupo: {filepath}")

        df = pd.read_csv(filepath)
        
        # Estrategia generalizada para identificar el ancla:
        # 1. Chequea si el grouper ha definido explícitamente una columna de ancla.
        # 2. Si no, busca la columna de distancia configurada y asume que el ancla tiene distancia 0.
        if LOCAL_ANCHOR_COL in df.columns:
            is_anchor = df[LOCAL_ANCHOR_COL] == True
        elif LOCAL_DISTANCE_COL in df.columns:
            is_anchor = np.isclose(df[LOCAL_DISTANCE_COL], 0.0, atol=1e-8)
        else:
            logger.error(f"El archivo carece de '{LOCAL_ANCHOR_COL}' o '{LOCAL_DISTANCE_COL}'.")
            raise ValueError("Formato de vecindario inválido. No se puede identificar el ancla.")

        anchor_df = df[is_anchor].copy()
        neighbors_df = df[~is_anchor].copy()

        if anchor_df.empty:
            logger.warning("No se encontró ninguna observación ancla en el grupo.")
        elif len(anchor_df) > 1:
            logger.warning(f"Se encontraron {len(anchor_df)} anclas (registros duplicados/idénticos).")
        else:
            logger.info(f"Grupo cargado con éxito: 1 ancla identificada y {len(neighbors_df)} vecinos.")

        return anchor_df, neighbors_df

    def plot_local_explanation(
        self, 
        explanation: pd.Series, 
        title: str, 
        filename: str,
        feature_values: Optional[pd.Series] = None
    ) -> Path:
        """
        Grafica la explicación local. Permite incluir los valores reales de las 
        características en el eje Y para facilitar la interpretación visual inmediata.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Ordenar por el valor absoluto del impacto
        exp_sorted = explanation.reindex(explanation.abs().sort_values(ascending=True).index)
        colors = ["#DD8452" if val < 0 else "#4C72B0" for val in exp_sorted]
        
        # Generar etiquetas dinámicas: si se pasan los valores reales, los concatenamos
        labels = exp_sorted.index
        if feature_values is not None:
            # Formatea la etiqueta como: "NombreCaracteristica = Valor"
            # Redondea valores numéricos si es necesario para mantener limpieza visual
            labels = []
            for feat in exp_sorted.index:
                val = feature_values.get(feat, 'N/A')
                if isinstance(val, float):
                    labels.append(f"{feat} = {val:.4g}")
                else:
                    labels.append(f"{feat} = {val}")
        
        ax.barh(labels, exp_sorted.values, color=colors, edgecolor="black", alpha=0.8) #type: ignore
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(exp1, exp2)

        if np.isnan(rho): #type: ignore
            return 0.0
            
        return float(rho) #type: ignore

    def compute_lipschitz(self, exp_anchor: np.ndarray, exp_neighbor: np.ndarray, distance: float) -> float:
        """
        Calcula la constante local de Lipschitz dado el impacto del ancla y vecino 
        escalado por la distancia generalizada entre ellos.
        """
        if distance <= 1e-8:
            return 0.0

        rho = self.compute_spearman(exp_anchor, exp_neighbor)
        d_exp = (1.0 - rho) / 2.0
        lipschitz_constant = d_exp / distance

        return float(lipschitz_constant)
    
    @staticmethod
    def plot_stability_scatter(stability_csv: Path) -> Path:
        """
        Grafica la métrica de estabilidad frente a la distancia agnóstica.
        Mapea dinámicamente cualquier métrica subyacente que haya sido 
        exportada bajo el nombre 'distance'.
        """
        if not stability_csv.exists():
            logger.error(f"No se encontró el archivo de estabilidad: {stability_csv}")
            raise FileNotFoundError(f"Archivo no encontrado: {stability_csv}")

        df = pd.read_csv(stability_csv)
        
        # Validación usando la columna de distancia agnóstica
        required_cols = ["distance", "spearman_rho", "lipschitz_constant"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"El archivo carece de la columna métrica requerida: {col}")
                raise ValueError(f"Archivo inválido. Columna faltante: {col}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sc1 = ax1.scatter(
            df["distance"], df["spearman_rho"], 
            c=df["lipschitz_constant"], cmap="viridis", 
            alpha=0.7, edgecolor="black", s=50
        )
        ax1.set_xlabel("Distancia Espacial (Similitud de Observaciones)")
        ax1.set_ylabel("Correlación de Spearman (Similitud de Interpretación)")
        ax1.set_title("Degradación de la Explicación vs Perturbación")
        ax1.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        ax1.grid(True, linestyle="--", alpha=0.4)
        cbar = plt.colorbar(sc1, ax=ax1)
        cbar.set_label("Constante de Lipschitz (Inestabilidad)")

        ax2.scatter(
            df["distance"], df["lipschitz_constant"], 
            color="#DD8452", alpha=0.7, edgecolor="black", s=50
        )
        ax2.set_xlabel("Distancia Espacial (Similitud de Observaciones)")
        ax2.set_ylabel("Constante de Lipschitz")
        ax2.set_title("Magnitud de la Inestabilidad (Lipschitz)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        tech_name = stability_csv.stem.split("_")[0][:1].upper() + stability_csv.stem.split("_")[0][1:]
        plt.suptitle(f"Análisis de Estabilidad Espacial - {tech_name}", fontsize=14, y=1.02)
        plt.tight_layout()

        FIGURES_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"stability_scatter_{stability_csv.stem}.png"
        out_path = FIGURES_XAI_DIR / out_filename
        
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Panel de dispersión de estabilidad generado en: {out_path.name}")
        return out_path

    @staticmethod
    def compute_stability_summary(stability_csv: Path) -> Path:
        """
        Lee los resultados de estabilidad y genera un reporte estadístico detallado
        incluyendo estadísticos de tendencia central, dispersión, extremos y 'Trust Scores'.
        """
        if not stability_csv.exists():
            logger.error(f"No se encontró el archivo de estabilidad: {stability_csv}")
            raise FileNotFoundError(f"Archivo no encontrado: {stability_csv}")

        df = pd.read_csv(stability_csv)
        cols_to_analyze = ["spearman_rho", "lipschitz_constant"]
        
        for col in cols_to_analyze:
            if col not in df.columns:
                logger.error(f"El archivo carece de la métrica {col}.")
                raise ValueError(f"Falta la columna {col}")

        stats_dict = {}
        n_total = len(df)
        
        # Iterar sobre las métricas para extraer estadísticas completas
        for col in cols_to_analyze:
            series = df[col]
            stats_dict[col] = {
                "count": n_total,
                "mean": series.mean(),
                "median": series.median(),
                "variance": series.var(),
                "std_dev": series.std(),
                "min": series.min(),
                "max": series.max(),
                "percentile_25": series.quantile(0.25),
                "percentile_75": series.quantile(0.75),
                "percentile_90": series.quantile(0.90),
                "percentile_95": series.quantile(0.95)
            }
            
        # Calcular Trust Scores (métricas de umbral)
        degraded_spearman = (df["spearman_rho"] < STABILITY_SPEARMAN_THRESHOLD).sum()
        severe_lipschitz = (df["lipschitz_constant"] > STABILITY_LIPSCHITZ_THRESHOLD).sum()
        
        stats_dict["spearman_rho"]["degradation_threshold"] = STABILITY_SPEARMAN_THRESHOLD
        stats_dict["spearman_rho"]["degradation_count"] = degraded_spearman
        stats_dict["spearman_rho"]["degradation_rate_%"] = (degraded_spearman / n_total) * 100
        
        stats_dict["lipschitz_constant"]["severe_instability_threshold"] = STABILITY_LIPSCHITZ_THRESHOLD
        stats_dict["lipschitz_constant"]["severe_instability_count"] = severe_lipschitz
        stats_dict["lipschitz_constant"]["severe_instability_rate_%"] = (severe_lipschitz / n_total) * 100

        summary_df = pd.DataFrame(stats_dict)
        
        RESULTS_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_XAI_DIR / f"summary_{stability_csv.stem}.csv"
        summary_df.to_csv(out_path, index=True, index_label="statistic")
        
        logger.info(f"Reporte estadístico exhaustivo guardado en: {out_path.name}")
        return out_path

    @staticmethod
    def plot_stability_distribution(stability_csv: Path) -> Path:
        """
        Genera una cuadrícula de visualizaciones con Histogramas (Densidad) y Boxplots
        para observar en detalle la distribución de las métricas de estabilidad.
        """
        if not stability_csv.exists():
            logger.error(f"No se encontró el archivo de estabilidad: {stability_csv}")
            raise FileNotFoundError(f"Archivo no encontrado: {stability_csv}")

        df = pd.read_csv(stability_csv)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [
            ("spearman_rho", "Correlación de Spearman (Similitud)", "blue", axes[0,0], axes[0,1]),
            ("lipschitz_constant", "Constante de Lipschitz (Inestabilidad)", "orange", axes[1,0], axes[1,1])
        ]
        
        for col, title, color, ax_hist, ax_box in metrics:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            # 1. Histograma + KDE
            ax_hist.hist(series, bins=20, color=color, alpha=0.6, density=True, edgecolor='black')
            try:
                # Se intenta graficar la curva de densidad (KDE)
                series.plot(kind="kde", ax=ax_hist, color="red", linewidth=2)
            except Exception:
                # Ocurre si la varianza es 0 (ej. todos los valores son iguales)
                pass
            ax_hist.set_title(f"Distribución - {title}")
            ax_hist.set_xlabel(title)
            ax_hist.set_ylabel("Densidad")
            ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

            # 2. Boxplot
            boxprops = dict(facecolor=color, color="black", alpha=0.7)
            medianprops = dict(color="red", linewidth=2)
            ax_box.boxplot(series, vert=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
            ax_box.set_title(f"Boxplot - {title}")
            ax_box.set_xlabel(title)
            ax_box.set_yticks([]) 
            ax_box.grid(axis='x', linestyle='--', alpha=0.5)

        plt.suptitle(f"Análisis de Distribución de Estabilidad Local", fontsize=15, y=1.02)
        plt.tight_layout()

        FIGURES_XAI_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"stability_dist_{stability_csv.stem}.png"
        out_path = FIGURES_XAI_DIR / out_filename
        
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Gráfico de distribuciones (Hist/Boxplot) generado en: {out_path.name}")
        return out_path