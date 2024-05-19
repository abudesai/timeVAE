import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(ROOT_DIR, "data")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

GEN_DATA_DIR = os.path.join(OUTPUTS_DIR, "gen_data")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
TSNE_DIR = os.path.join(OUTPUTS_DIR, "tsne")

SRC_DIR = os.path.join(ROOT_DIR, "src")

CONFIG_DIR = os.path.join(SRC_DIR, "config")
CFG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")
HYPERPARAMETERS_FILE_PATH = os.path.join(CONFIG_DIR, "hyperparameters.yaml")


# MODEL ARTIFACTS
SCALER_FILE_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
