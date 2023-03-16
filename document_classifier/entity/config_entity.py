from dataclasses import dataclass
import os
from document_classifier.utils.main_utils import MainUtils
from document_classifier.constant import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.utils = MainUtils()
        self.data_ingestion_artifacts_dir: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.train_data_download_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, S3_TRAIN_DATA_FILE_NAME) 
        self.eval_data_download_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, S3_EVAL_DATA_FILE_NAME) 
        self.unzip_data_folder_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.idx2label_file_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, INDEX_TO_LABEL_FILE_NAME)
        self.label2idx_file_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, LABEL_TO_IDX_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_artifacts_dir: str = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.train_dataloader_filepath: str = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR, TRAIN_DATALOADER_FILE_NAME)
        self.test_dataloader_filepath: str = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR, TEST_DATALOADER_FILE_NAME)
        self.eval_dataloader_filepath: str = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR, EVAL_DATALOADER_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_artifacts_dir: str = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.trained_model_path: str = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_DIR_NAME)


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.model_evaluation_artifacts_dir: str = os.path.join(ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.trained_model_path: str = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_DIR_NAME)
        self.s3_local_model_path: str = TRAINED_MODEL_DIR_NAME


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.s3_model_path: str = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_DIR_NAME)