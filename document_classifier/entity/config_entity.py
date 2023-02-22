from dataclasses import dataclass
from from_root import from_root
import os
from document_classifier.utils.main_utils import MainUtils
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.constant import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.utils = MainUtils()
        self.data_ingestion_artifacts_dir: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.data_download_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, S3_DATA_FILE_NAME) 
        self.unzip_data_folder_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, UNZIP_FOLDER_NAME)
        self.train_dataset_file_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, TRAIN_DATA_PICKLE_FILE_NAME)
        self.test_dataset_file_path: str = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, TEST_DATA_PICKLE_FILE_NAME)