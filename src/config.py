"""
Configuración central del proyecto: semillas, rutas, parámetros base y sistema de logging.
Todos los módulos deben importar desde aquí para evitar valores hardcodeados.
"""

import logging
import sys
from pathlib import Path

# ===========================================================================
# 1. Rutas base
# ===========================================================================
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODELS_DIR  = OUTPUT_DIR / "models"
SCALERS_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR    = OUTPUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH   = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
CLEAN_DATA_PATH = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015_no_outliners.csv"

# ===========================================================================
# 2. Configuración del Logger
# ===========================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Instancia y configura un logger formal para el módulo que lo solicite.
    
    El formato incluye:
    - Fecha y hora exacta.
    - Nivel de severidad (INFO, WARNING, ERROR, etc.).
    - Origen: Archivo, Función y Línea exacta de código.
    - Mensaje.
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        
        # Ejemplo: [2026-06-14 10:15:30] [INFO] [trainer.run_cv:145] - Entrenando fold 1...
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s.py:%(lineno)d][%(funcName)s] - %(message)s",
            datefmt="%d/%m %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # En terminal solo mostramos INFO o superior
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(LOGS_DIR / "pipeline_execution.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        logger.propagate = False

    return logger

# ===========================================================================
# 3. Semillas
# ===========================================================================
GLOBAL_SEED      = 314159           # semilla maestra
SPLIT_SEED       = 271828           # estratificación holdout
CV_SEED          = 141421           # generación de folds
DT_SEED          = 7                # árbol de decisión
XGBOOST_SEED     = 13               # XGBoost
THRESHOLD_SEED   = 66616            # barrido de umbral

# ===========================================================================
# 4. Proporciones del split estratificado y Validación Cruzada
# ===========================================================================
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.20
TUNE_SIZE  = 0.10   # holdout para tuning de hiperparámetros y umbral
N_FOLDS    = 5      # CV estratificado sobre el conjunto de entrenamiento

# ===========================================================================
# 5. Variable objetivo y columnas
# ===========================================================================
TARGET_COL = "Diabetes_binary"

# Variables continuas/discretas que serán estandarizadas.
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

# Variables ordinales que se escalan exclusivamente para los métodos XAI.
ORDINAL_COLS = ["Age", "Education", "Income", "GenHlth"]

# ===========================================================================
# 6. Parámetros base de los modelos
# ===========================================================================
DT_BASE_PARAMS = {
    "criterion"   : "entropy",
    "max_depth"   : 10,
    "random_state": DT_SEED,
}

XGBOOST_BASE_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 6,
    "learning_rate"   : 0.1,
    "subsample"       : 0.8,
    "eval_metric"     : "aucpr",
    "objective"       : "binary:logistic",
    "random_state"    : XGBOOST_SEED,
}

# ===========================================================================
# 7. Umbral de clasificación
# ===========================================================================
DEFAULT_THRESHOLD    = 0.5
THRESHOLD_SEARCH_MIN = 0.01
THRESHOLD_SEARCH_MAX = 0.99
THRESHOLD_STEPS      = 120

# ===========================================================================
# 8. Espacios de búsqueda para Optimización Bayesiana (Optuna)
# ===========================================================================
OPTUNA_TRIALS    = 1500
OPTUNA_PATIENCE  = 100
OPTUNA_TOLERANCE = 0.001

def dt_param_space(trial) -> dict:
    """Espacio de búsqueda hiperparamétrica estructurado para Decision Tree."""
    return {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),        
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.005),
        "random_state": DT_SEED,
    }

def xgb_param_space(trial) -> dict:
    """Espacio de búsqueda hiperparamétrica para XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),        
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
        
        "tree_method": "hist",
        "max_bin": trial.suggest_categorical("max_bin", [32, 64, 128, 256, 512]),
        "device": "cuda", 
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": XGBOOST_SEED,
    }