"""
Módulo de Optimización de Hiperparámetros Generalizado.

Utiliza Optuna (Optimización Bayesiana) para encontrar los mejores 
parámetros de cualquier clasificador mediante validación cruzada.
Guarda el historial completo de ensayos (trials) para depuración,
y redirige los logs nativos de Optuna a un archivo específico.
"""
import logging
import json
import optuna
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional

from src.evaluator import Evaluator
from src.config import RESULTS_DIR, LOGS_DIR, get_logger

logger = get_logger(__name__)


class ConvergenceEarlyStoppingCallback:
    """
    Criterio de parada por convergencia matemática para Optuna.
    Detiene el estudio si el valor no mejora tras 'patience' ensayos.
    """
    def __init__(self, patience: int = 15, tolerance: float = 0.001) -> None:
        self.patience = patience
        self.tolerance = tolerance
        self.no_improvement_count = 0
        self.best_score = -np.inf

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        current_best = study.best_value

        if (current_best - self.best_score) > self.tolerance:
            self.best_score = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            logger.info(
                f"Convergencia matemática alcanzada. "
                f"Sin mejora > {self.tolerance} durante {self.patience} ensayos. Deteniendo estudio."
            )
            study.stop()


class HyperparameterOptimizer:
    """
    Optimizador agnóstico de modelos mediante validación cruzada.
    """
    def __init__(
        self, 
        study_name: str, 
        direction: str = "maximize", 
        n_trials: int = 50,
        dataset_tag: str = "full"
    ) -> None:
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.dataset_tag = dataset_tag
        
        self._configure_optuna_logger()
        
        self.study = optuna.create_study(
            direction=self.direction, 
            study_name=self.study_name
        )
        logger.debug(f"Optimizador instanciado: {self.study_name} | Dataset: {self.dataset_tag}")

    def _configure_optuna_logger(self) -> None:
        """
        Intercepta y enruta los logs nativos de Optuna exclusivamente hacia un archivo.
        """
        optuna.logging.disable_default_handler()
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        
        optuna_logger.handlers.clear()
        
        log_path = LOGS_DIR / f"optuna_{self.study_name}_{self.dataset_tag}.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [Ensayo %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        optuna_logger.addHandler(file_handler)

    def optimize_cv(
        self,
        model_class: Any,
        param_space_func: Callable[[optuna.Trial], dict],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splits: list,
        preprocessor_class: Any,
        fit_kwargs_func: Optional[Callable[[pd.Series], dict]] = None,
        metric: str = "prauc",
        patience: int = 100,
        tolerance: float = 0.001
    ) -> dict:
        """
        Ejecuta la optimización evaluando cada combinación de parámetros 
        a través de todos los pliegues de la validación cruzada.
        """
        logger.info(f"Iniciando estudio '{self.study_name}' con máximo de {self.n_trials} ensayos...")

        def objective(trial: optuna.Trial) -> float:
            try:
                params = param_space_func(trial)
                fold_scores = []
                
                for fold_idx, (idx_tr, idx_val) in enumerate(cv_splits):
                    X_fold_tr, y_fold_tr = X_train.iloc[idx_tr], y_train.iloc[idx_tr]
                    X_fold_val, y_fold_val = X_train.iloc[idx_val], y_train.iloc[idx_val]
                    
                    prep = preprocessor_class(
                        fold_id=f"opt_{trial.number}_f{fold_idx}", 
                        dataset_tag=self.dataset_tag
                    )
                    X_tr_scaled = prep.fit_transform(X_fold_tr)
                    X_val_scaled = prep.transform(X_fold_val)
                    
                    model = model_class(**params)
                    fit_kwargs = fit_kwargs_func(y_fold_tr) if fit_kwargs_func else {}
                    
                    model.fit(X_tr_scaled, y_fold_tr, **fit_kwargs)
                    
                    evaluator = Evaluator()
                    metrics = evaluator.evaluate(model, X_val_scaled, y_fold_val)
                    fold_scores.append(metrics[metric])
                
                return float(np.mean(fold_scores))
            
            except Exception as e:
                logger.warning(f"Ensayo {trial.number} podado (fallo interno): {e}")
                raise optuna.TrialPruned()

        early_stopping = ConvergenceEarlyStoppingCallback(
            patience=patience, 
            tolerance=tolerance
        )

        try:
            self.study.optimize(objective, n_trials=self.n_trials, callbacks=[early_stopping])
            
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            logger.info(f"Estudio '{self.study_name}' completado. Mejor {metric.upper()}: {best_score:.4f}")
            self._save_results(best_params, best_score, metric)
            
            return best_params
            
        except ValueError as e:
            logger.error(f"Fallo en la optimización: {e}")
            raise
        except Exception as e:
            logger.error(f"Error crítico durante el proceso de Optuna: {e}")
            raise

    def _save_results(self, best_params: dict, best_score: float, metric: str) -> None:
        """
        Guarda los mejores parámetros (JSON) y el historial completo (CSV).
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        best_file = RESULTS_DIR / f"{self.study_name}_best_params_{self.dataset_tag}.json"
        payload = {
            "metric_optimized": metric,
            "best_score": best_score,
            "best_params": best_params
        }
        
        try:
            with open(best_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            logger.info(f"Mejores parámetros guardados exitosamente en: {best_file.name}")
        except Exception as e:
            logger.error(f"No se pudo guardar el archivo JSON de parámetros óptimos: {e}")
            raise
            
        try:
            trials_df = self.study.trials_dataframe()
            trials_file = RESULTS_DIR / f"{self.study_name}_trials_history_{self.dataset_tag}.csv"
            trials_df.to_csv(trials_file, index=False)
            logger.info(f"Historial tabular de ensayos guardado en: {trials_file.name}")
        except Exception as e:
            logger.error(f"No se pudo exportar el historial de Optuna a CSV: {e}")
            raise