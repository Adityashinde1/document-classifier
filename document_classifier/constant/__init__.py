import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'document_classifier.log' 

BUCKET_NAME = 'document-classification-io-files'
S3_TRAIN_DATA_FILE_NAME = 'document_train_dataset.zip'
S3_EVAL_DATA_FILE_NAME = 'document_eval_dataset.zip'


DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
TRAIN_DATA_UNZIP_FOLDER_NAME = 'document_train_dataset'
EVAL_DATA_UNZIP_FOLDER_NAME = 'document_eval_dataset'
TEST_SIZE = 0.2
LABEL_TO_IDX_FILE_NAME = 'label2idx.json'
INDEX_TO_LABEL_FILE_NAME = 'idx2label.json'


DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRAIN_DATALOADER_FILE_NAME = 'train_dataloader.pkl'
TEST_DATALOADER_FILE_NAME = 'test_dataloader.pkl'
EVAL_DATALOADER_FILE_NAME = 'eval_dataloader.pkl'


MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
LEARNING_RATE = 5e-5
EPOCHS = 20
TRAINED_MODEL_DIR_NAME = 'saved_model/'
SAVED_MODEL_DIR = 'saved_model'

MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'

BEST_MODEL_DIR = 'best_model'

APP_HOST = "0.0.0.0"
APP_PORT = 8080