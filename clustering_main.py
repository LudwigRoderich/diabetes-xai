"""
Punto de entrada de la etapa de Clustering del proyecto MM-700.
Orquesta la optimización por silueta y la extracción de radios de clúster.
"""
import argparse
import json
import sys
import traceback
import pandas as pd

from src.config import get_logger, DEFAULT_K_CLUSTERS, CLUSTER_SEED, LOGS_DIR, CLUSTERS_DIR, MAX_CLUSTER_SAMPLE_SIZE
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.persistence import Persistence
from src.clusterizer import Clusterizer

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa de Clustering con optimización geométrica para XAI.")
    parser.add_argument("--dataset", choices=["full", "clean", "both"], default="both")
    parser.add_argument("--algorithm", choices=["clara", "kmeans"], default="clara")
    parser.add_argument("--k-clusters", type=int, default=DEFAULT_K_CLUSTERS)
    parser.add_argument("--optimize-k", action="store_true", default=False, help="Habilita optimización por Índice de Silueta.")
    parser.add_argument("--k-range", type=int, nargs=2, default=[5, 15], help="Rango [mínimo, máximo] para optimización.")
    parser.add_argument("--seed", type=int, default=CLUSTER_SEED)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--join", action="store_true", default=False, help="Si se activa, se unen los datos escogidos con su clúster ya asignado")
    return parser.parse_args()

def run_clustering(dataset_tag: str, args: argparse.Namespace) -> None:
    use_clean = dataset_tag == "clean"
    logger.info(f"Iniciando flujo de clustering | Dataset: {dataset_tag.upper()}")


    try:
        loader = DataLoader(use_clean=use_clean).load().split()
        try:
            prep = Preprocessor.load(fold_id="final", dataset_tag=dataset_tag)
        except FileNotFoundError:
            logger.error(f"Preprocesador 'final' ausente para {dataset_tag}. Ejecuta el pipeline de clasificación primero.")
            sys.exit(1)

        if args.split == "val":
            X_raw = loader.X_val
        elif args.split == "train":
            X_raw = loader.X_train
        else:
            X_raw = pd.concat([loader.X_train, loader.X_tune, loader.X_val])

        if X_raw is None or X_raw.empty:
            logger.error("El conjunto de datos seleccionado no contiene registros.")
            return

        X_scaled = prep.transform(X_raw, xai_mode=False)
        clusterizer = Clusterizer(algorithm=args.algorithm, k=args.k_clusters, seed=args.seed)

        if args.optimize_k:
            k_min, k_max = args.k_range
            chosen_k = clusterizer.optimize_k(X_scaled=X_scaled, X_raw=X_raw, k_range=(k_min, k_max), max_sample_size=MAX_CLUSTER_SAMPLE_SIZE)
        else:
            chosen_k = args.k_clusters

        labels_series, cluster_metadata = clusterizer.fit_predict(X_scaled=X_scaled, X_raw=X_raw)

        df_assignments = pd.DataFrame({
            "original_index": X_scaled.index,
            "cluster_label": labels_series.values
        })

        metadata_path = CLUSTERS_DIR / f"cluster_metadata_{dataset_tag}_k{chosen_k}.json"
        with open(metadata_path, "w") as f:
            json.dump(cluster_metadata, f, indent=4)
        logger.info(f"Metadatos geométricos guardados en {metadata_path}")

        Persistence.save_cluster_assignments(
            df_assignments=df_assignments,
            algorithm=args.algorithm,
            k=chosen_k,
            dataset_tag=dataset_tag,
            seed=args.seed
        )
        
        Persistence.save_cluster_model(
            model=clusterizer.get_model(),
            algorithm=args.algorithm,
            k=chosen_k,
            dataset_tag=dataset_tag,
            seed=args.seed
        )
        
        logger.info(f"Proceso concluido exitosamente para {dataset_tag}.")

    except Exception as e:
        error_msg = f"Error crítico en run_clustering para {dataset_tag}:\n{traceback.format_exc()}"
        logger.error(error_msg)
        error_log_path = LOGS_DIR / f"error_{dataset_tag}.log"
        with open(error_log_path, "w") as f:
            f.write(error_msg)

if __name__ == "__main__":
    args = parse_args()
    tags = ["full", "clean"] if args.dataset == "both" else [args.dataset]

    logger.info("=== Control de Clustering e Inferencia de Radios ===")
    for tag in tags:
        run_clustering(tag, args)
    logger.info("=== Finalización del Proceso Geométrico ===")