"""
Configuración central del proyecto: semillas, rutas y parámetros base.
Todos los módulos deben importar desde aquí para evitar valores hardcodeados.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODELS_DIR  = OUTPUT_DIR / "models"
SCALERS_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

RAW_DATA_PATH     = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
CLEAN_DATA_PATH   = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015_no_outliners.csv"   # sin outliers

# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------
GLOBAL_SEED      = 314159           # semilla maestra
SPLIT_SEED       = 271828           # estratificación holdout
CV_SEED          = 141421           # generación de folds
DT_SEED          = 7                # árbol de decisión
XGBOOST_SEED     = 13               # XGBoost
THRESHOLD_SEED   = 66616             # barrido de umbral (si aplica aleatoriedad)

# ---------------------------------------------------------------------------
# Proporciones del split estratificado
# ---------------------------------------------------------------------------
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.20
TUNE_SIZE  = 0.10   # holdout para tuning de hiperparámetros y umbral

# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
N_FOLDS = 5         # CV estratificado sobre el conjunto de entrenamiento

# ---------------------------------------------------------------------------
# Variable objetivo y columnas
# ---------------------------------------------------------------------------
TARGET_COL = "Diabetes_binary"

# Variables continuas/discretas que serán estandarizadas.
# Las binarias (0/1) se dejan sin escalar.
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

# Variables ordinales que se escalan exclusivamente para los métodos XAI.
ORDINAL_COLS = ["Age", "Education", "Income", "GenHlth"]

# ---------------------------------------------------------------------------
# Parámetros base de los modelos (se ajustarán en la fase de tuning)
# ---------------------------------------------------------------------------
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
    "use_label_encoder": False,
    "eval_metric"     : "aucpr",
    "random_state"    : XGBOOST_SEED,
}

# ---------------------------------------------------------------------------
# Umbral de clasificación (valor inicial; se optimiza sobre TUNE set)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD    = 0.5
THRESHOLD_SEARCH_MIN = 0.10
THRESHOLD_SEARCH_MAX = 0.90
THRESHOLD_STEPS      = 81     # paso de 0.01 sobre [0.10, 0.90]