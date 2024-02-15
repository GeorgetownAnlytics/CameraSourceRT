import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_DIR = os.path.join(ROOT_DIR, "src")

CONFIG_DIR = os.path.join(SRC_DIR, "config")

RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")

VISION_DATA_DIR = os.path.join(ROOT_DIR, "datasets", "Vision_data")

CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

HYPERPARAMETERS_FILE = os.path.join(CONFIG_DIR, "hyperparameters.json")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TRAINING_LOGS_FILE = os.path.join(LOGS_DIR, "training.log")

MODEL_INPUTS_OUTPUTS_DIR = os.path.join(ROOT_DIR, "model_inputs_outputs")

INPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "inputs")

TRAIN_DIR = os.path.join(INPUTS_DIR, "train")

TEST_DIR = os.path.join(INPUTS_DIR, "test")

VALIDATION_DIR = os.path.join(INPUTS_DIR, "validation")

OUTPUTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "outputs")

PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "predictions.csv")


RUN_ALL_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, "run_all_predictions")

ERRORS_DIR = os.path.join(OUTPUTS_DIR, "errors")

MODEL_ARTIFACTS_DIR = os.path.join(MODEL_INPUTS_OUTPUTS_DIR, "artifacts")

DATA_SPLIT_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "data_split")

PREDICTOR_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "predictor")

MODEL_DATA_FILE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "model_data.joblib")

CHECKPOINTS_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "checkpoints")

RUN_ALL_ARTIFACTS_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "run_all_artifacts")
