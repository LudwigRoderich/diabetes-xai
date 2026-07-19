"""
Configuración central del proyecto: semillas, rutas, parámetros base y sistema de logging.
Todos los módulos deben importar desde aquí para evitar valores hardcodeados.
"""

from datetime import datetime
import logging
import sys
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 1. Rutas base

ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODELS_DIR  = OUTPUT_DIR / "models"
SCALERS_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR    = OUTPUT_DIR / "logs"
XAI_DIR     = OUTPUT_DIR / "xai"

RESULTS_CLF_DIR   = RESULTS_DIR / "binary_classification"
RESULTS_XAI_DIR   = RESULTS_DIR / "xai"
RESULTS_GROUP_DIR = RESULTS_DIR / "grouping"

FIGURES_CLF_DIR   = FIGURES_DIR / "binary_classification"
FIGURES_XAI_DIR   = FIGURES_DIR / "xai"
FIGURES_GROUP_DIR = FIGURES_DIR / "grouping"

LOGS_CLF_DIR      = LOGS_DIR / "binary_classification"
LOGS_XAI_DIR      = LOGS_DIR / "xai"
LOGS_GROUP_DIR    = LOGS_DIR / "grouping"

for d in [
    MODELS_DIR, SCALERS_DIR,
    RESULTS_CLF_DIR, RESULTS_XAI_DIR, RESULTS_GROUP_DIR,
    FIGURES_CLF_DIR, FIGURES_XAI_DIR, FIGURES_GROUP_DIR,
    LOGS_CLF_DIR, LOGS_XAI_DIR, LOGS_GROUP_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH   = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
CLEAN_DATA_PATH = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015_no_outliners.csv"




# 2. Configuración del Logger
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
def get_logger(name: str) -> logging.Logger:
    """
    Instancia y configura un logger formal. Redirige dinámicamente el log a 
    la subcarpeta correspondiente según el nombre del módulo o flujo.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    if "xai" in name.lower():
        log_file = LOGS_XAI_DIR / "xai_pipeline.log"
    elif "group" in name.lower():
        log_file = LOGS_GROUP_DIR / "grouping_pipeline.log"
    else:
        log_file = LOGS_CLF_DIR / "binary_classification.log"

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file, encoding="utf-8")

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


# 3. Semillas

GLOBAL_SEED      = 314159           # semilla maestra
SPLIT_SEED       = 271828           # estratificación holdout
CV_SEED          = 141421           # generación de folds
DT_SEED          = 7                # árbol de decisión
XGBOOST_SEED     = 13               # XGBoost
THRESHOLD_SEED   = 66616            # barrido de umbral
PFI_SEED         = 123456           # Permutation Feature Importance

# 4. Proporciones del split estratificado y Validación Cruzada

TRAIN_SIZE = 0.70
VAL_SIZE   = 0.20
TUNE_SIZE  = 0.10
N_FOLDS    = 5

# 5. Variable objetivo y columnas

TARGET_COL = "Diabetes_binary"
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"] # Variables continuas/discretas que serán estandarizadas.
ORDINAL_COLS = ["Age", "Education", "Income", "GenHlth"] # Variables ordinales que se escalan exclusivamente para los métodos XAI.

# 6. Parámetros base de los modelos

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

# 7. Umbral de clasificación

DEFAULT_THRESHOLD    = 0.5
THRESHOLD_SEARCH_MIN = 0.01
THRESHOLD_SEARCH_MAX = 0.99
THRESHOLD_STEPS      = 120
THRESH_METRIC_CHOICES = (
    "balanced_accuracy", 
    "f0.5", 
    "f1", 
    "f2", 
    "gmean", 
    "mcc", 
    "precision", 
    "recall", 
    "youden_j"
)

# 8. Espacios de búsqueda para Optimización Bayesiana (Optuna)

OPTUNA_TRIALS    = 1500
OPTUNA_PATIENCE  = 100
OPTUNA_TOLERANCE = 0.001
OPTIMIZATION_METRIC_CHOICES = ("prauc", "precision", "recall", "f1", "f2", "mcc")

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

# 9. Registro central de modelos.
MODEL_CONFIGS = {
    "dt": {
        "display_name": "Decision Tree",
        "estimator_class": DecisionTreeClassifier,
        "base_params": DT_BASE_PARAMS,
        "fixed_params": {},
        "param_space": dt_param_space,
        "color": "#4C72B0",
        "uses_sample_weight": True,
        "uses_scale_pos_weight": False,
    },
    "xgb": {
        "display_name": "XGBoost",
        "estimator_class": XGBClassifier,
        "base_params": XGBOOST_BASE_PARAMS,
        "fixed_params": {
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
        },
        "param_space": xgb_param_space,
        "color": "#DD8452",
        "uses_sample_weight": False,
        "uses_scale_pos_weight": True,
    },
}

MODEL_TAGS = tuple(MODEL_CONFIGS.keys())
DATASET_TAGS = ("full", "clean")
PARTITION_TAGS = ("train", "val", "tune")

# 10. Configuración XAI

# 10.1 Permutation Feature Importance (PFI)
PFI_NJOBS     = -1
PFI_N_REPEATS = 50
PFI_SCORING    = "average_precision" 
PFI_MAX_SAMPLES = 1.0