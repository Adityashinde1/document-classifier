import os
from from_root import from_root
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join(from_root(), "artifacts", TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'document_classifier.log' 

BUCKET_NAME = 'document-classification-io-files'
S3_DATA_FILE_NAME = 'document_dataset.zip'


DATA_INGESTION_ARTIFACTS_DIR = 'dataIngestionArtifacts'
UNZIP_FOLDER_NAME = 'data'
TEST_SIZE = 0.09
TRAIN_DATA_PICKLE_FILE_NAME = 'train_data.pkl'
TEST_DATA_PICKLE_FILE_NAME = 'test_data.pkl'