"""
Módulo optimizado para el agrupamiento de datos (Clustering) del proyecto MM-700.
Soporta optimización por Silueta y cálculo de radios de vecindad para análisis XAI.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import gower
import matplotlib.pyplot as plt
import os
import json
import traceback

from src.config import get_logger, CLUSTER_SEED, CLUSTER_TOLERANCE, CLARA_SAMPLE_FRAC, CLARA_MAX_SAMPLE, CLARA_N_SAMPLES, LOGS_DIR, FIGURES_DIR

logger = get_logger(__name__)

class Clusterizer:
    def __init__(self, algorithm: str = "kmedoids", k: int = 3, seed: int = CLUSTER_SEED, tolerance: float = CLUSTER_TOLERANCE):
        self.algorithm = algorithm.lower()
        self.k = k
        self.seed = seed
        self.tolerance = tolerance
        self.model = None
        self.dist_matrix = None

        self.clara_frac = CLARA_SAMPLE_FRAC
        self.clara_max_sample = CLARA_MAX_SAMPLE
        self.clara_n_samples = CLARA_N_SAMPLES

    def _run_clara_core(self, X_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        n_total = len(X_raw)
        sample_size = int(n_total * self.clara_frac)
        sample_size = min(sample_size, self.clara_max_sample)
        sample_size = max(sample_size, self.k * 2) 
        
        rng = np.random.default_rng(self.seed)
        
        best_cost = float('inf')
        best_labels = None
        best_medoids_idx = None
        
        clara_log = {
            "n_total": n_total,
            "sample_size": sample_size,
            "n_iterations": self.clara_n_samples,
            "iterations_data": []
        }
        
        logger.info(f"Iniciando CLARA: {self.clara_n_samples} iteraciones con muestras de tamaño {sample_size}")
        
        for i in range(self.clara_n_samples):
            try:
                sample_indices = rng.choice(X_raw.index, size=sample_size, replace=False)
                X_sample = X_raw.loc[sample_indices]
                
                dist_matrix_sample = gower.gower_matrix(X_sample)
                
                kmedoids_sample = KMedoids(n_clusters=self.k, metric='precomputed', random_state=self.seed + i)
                kmedoids_sample.fit(dist_matrix_sample)
                
                sample_medoid_positions = kmedoids_sample.medoid_indices_
                current_medoids_idx = sample_indices[sample_medoid_positions]
                X_medoids = X_raw.loc[current_medoids_idx]
                
                dist_all_to_medoids = gower.gower_matrix(X_raw, X_medoids)
                
                labels = np.argmin(dist_all_to_medoids, axis=1)
                min_distances = np.min(dist_all_to_medoids, axis=1)
                cost = float(np.mean(min_distances))
                
                logger.debug(f"CLARA Iteración {i+1}/{self.clara_n_samples} | Costo Gower: {cost:.4f}")
                
                clara_log["iterations_data"].append({
                    "iteration": i + 1,
                    "cost": cost,
                    "is_best": bool(cost < best_cost)
                })
                
                if cost < best_cost:
                    best_cost = cost
                    best_labels = labels
                    best_medoids_idx = current_medoids_idx
                    self.dist_all_to_medoids = dist_all_to_medoids
                    
            except Exception as e:
                logger.error(f"Error en iteración {i+1} de CLARA: {str(e)}")
                clara_log["iterations_data"].append({"iteration": i + 1, "error": str(e), "traceback": traceback.format_exc()})
        
        log_path = LOGS_DIR / f"clara_execution_k{self.k}.json"
        with open(log_path, "w") as f:
            json.dump(clara_log, f, indent=4)
                
        return best_medoids_idx, best_labels, best_cost #type: ignore
    
    def optimize_k(self, X_scaled: pd.DataFrame, X_raw: pd.DataFrame | None = None, k_range=(2, 6), max_sample_size: int = 3000) -> int:
        logger.info(f"Optimizando cantidad de clústeres en el rango {k_range}...")
        
        eval_sample_size = min(max_sample_size, len(X_scaled))
        best_k = k_range[0]
        best_score = -1.0
        
        optimization_log = {
            "algorithm": self.algorithm,
            "k_range": k_range,
            "eval_sample_size": eval_sample_size,
            "results": []
        }
        
        k_values = []
        silhouette_scores = []
        
        for k_test in range(k_range[0], k_range[1] + 1):
            self.k = k_test
            try:
                if self.algorithm == "clara":
                    if X_raw is None:
                        raise ValueError("Para optimizar K con CLARA, se requiere el DataFrame original (X_raw) para calcular distancias Gower.")
                    
                    _, labels, _ = self._run_clara_core(X_raw)
                    labels_series = pd.Series(labels, index=X_raw.index)
                    eval_idx = X_raw.sample(n=eval_sample_size, random_state=self.seed).index
                    score = silhouette_score(X_scaled.loc[eval_idx], labels_series.loc[eval_idx])
                else:
                    model = KMeans(n_clusters=k_test, random_state=self.seed, tol=self.tolerance, n_init='auto')
                    labels = model.fit_predict(X_scaled)
                    eval_idx = X_scaled.sample(n=eval_sample_size, random_state=self.seed).index
                    score = silhouette_score(X_scaled.loc[eval_idx], labels[eval_idx.get_indexer_for(X_scaled.index)])
                    
                logger.info(f"Evaluación: k={k_test} | Silhouette Score: {score:.4f}")
                
                k_values.append(k_test)
                silhouette_scores.append(float(score))
                optimization_log["results"].append({"k": k_test, "silhouette_score": float(score)})
                
                if score > best_score:
                    best_score = score
                    best_k = k_test
            except Exception as e:
                logger.error(f"Error evaluando k={k_test}: {str(e)}")
                optimization_log["results"].append({"k": k_test, "error": str(e)})
                
        self.k = best_k
        optimization_log["best_k"] = self.k
        optimization_log["best_score"] = float(best_score)
        
        logger.info(f"Configuración seleccionada: k={self.k} (Silueta Máxima: {best_score:.4f})")

        logs_path = LOGS_DIR / "k_optimization_metrics.json"
        with open(logs_path, "w") as f:        
            json.dump(optimization_log, f, indent=4)
            
        if k_values and silhouette_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
            plt.axvline(x=self.k, color='r', linestyle='--', label=f'Óptimo (k={self.k})')
            plt.title('Optimización de $K$ mediante Índice de Silueta')
            plt.xlabel('Número de Clústeres ($K$)')
            plt.ylabel('Índice de Silueta')
            plt.xticks(range(min(k_values), max(k_values) + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            fig_path = FIGURES_DIR / "silhouette_optimization.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            logger.info(f"Gráfico de optimización guardado en {fig_path}")
            
        return self.k

    def fit_predict(self, X_scaled: pd.DataFrame, X_raw: pd.DataFrame | None = None) -> tuple[pd.Series, dict]:
        """Ajusta el modelo definitivo usando CLARA y calcula los radios de vecindad."""
        cluster_metadata = {}
        
        if self.algorithm == "clara":
            if X_raw is None:
                raise ValueError("CLARA con Gower requiere el DataFrame original (X_raw).")
                
            best_medoids_idx, labels, best_cost = self._run_clara_core(X_raw)
            self.best_medoid_indices = best_medoids_idx
            
            # Cálculo de los radios (N, K)
            for cluster_id in range(self.k):
                member_mask = (labels == cluster_id)
                distances_to_medoid = self.dist_all_to_medoids[member_mask, cluster_id]
                
                cluster_metadata[str(cluster_id)] = {
                    "medoid_original_index": int(best_medoids_idx[cluster_id]),
                    "radius_max": float(np.max(distances_to_medoid)) if len(distances_to_medoid) > 0 else 0.0,
                    "radius_mean": float(np.mean(distances_to_medoid)) if len(distances_to_medoid) > 0 else 0.0,
                    "cluster_size": int(np.sum(member_mask)),
                    "clara_cost": float(best_cost)
                }
                
        elif self.algorithm == "kmeans":
            self.model = KMeans(n_clusters=self.k, random_state=self.seed, tol=self.tolerance, n_init='auto')
            labels = self.model.fit_predict(X_scaled)
            centroids = self.model.cluster_centers_
            
            cluster_metadata = {}
            for cluster_id in range(self.k):
                member_data = X_scaled.iloc[labels == cluster_id].values
                centroid = centroids[cluster_id]
                # Distancia euclidiana al centroide
                distances_to_centroid = np.linalg.norm(member_data - centroid, axis=1)
                
                cluster_metadata[str(cluster_id)] = {
                    "medoid_original_index": None, # No aplica a centroides puros
                    "radius_max": float(np.max(distances_to_centroid)) if len(distances_to_centroid) > 0 else 0.0,
                    "radius_mean": float(np.mean(distances_to_centroid)) if len(distances_to_centroid) > 0 else 0.0,
                    "cluster_size": int(len(member_data))
                }
        else:
            raise ValueError(f"Algoritmo '{self.algorithm}' no reconocido.")

        labels_series = pd.Series(labels, index=X_scaled.index, name="cluster_label")
        return labels_series, cluster_metadata


    def get_model(self):
        return self.model