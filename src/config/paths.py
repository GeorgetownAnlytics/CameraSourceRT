import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_DIR = os.path.join(ROOT_DIR, "src")

CONFIG_DIR = os.path.join(SRC_DIR, "config")

RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")

VISION_DATA_DIR = os.path.join(ROOT_DIR, "datasets", "Vision_data")

SPLIT_DATA_DIR = os.path.join(VISION_DATA_DIR, "split")

CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

HPT_FILE = os.path.join(CONFIG_DIR, "HPT.json")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TRAINING_LOGS_FILE = os.path.join(LOGS_DIR, "training.log")

MODEL_INPUTS_OUTPUTS_DIR = os.path.join(ROOT_DIR, "model_inputs_outputs")

INPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "inputs")

OUTPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "outputs")

MODEL_ARTIFACTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "artifacts")

PREDICTOR_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "predictor")

PREDICTOR_FILE_PATH = os.path.join(PREDICTOR_DIR, "predictor.joblib")

MODEL_DATA_FILE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "model_data.joblib")

CHECKPOINTS_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "checkpoints")

RUN_ALL_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "run_all_outputs")
