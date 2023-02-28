import os
import sys
import logging
from zipfile import ZipFile, Path
from document_classifier.utils.main_utils import MainUtils
from document_classifier.constant import *
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.exception import DocumentClassifierException
from document_classifier.entity.config_entity import DataIngestionConfig
from document_classifier.entity.artifacts_entity import DataIngestionArtifacts

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, s3_operation: S3Operation) -> None:
        self.data_ingestion_config = data_ingestion_config
        self.s3_operation = s3_operation
        self.utils = MainUtils()


    def get_data_from_s3(self, bucket_file_name: str, bucket_name: str, output_filepath: str) -> zip:
        try:
            logger.info("Entered the get_data_from_s3 method of Data ingestion class")
            self.s3_operation.read_data_from_s3(bucket_file_name, bucket_name, output_filepath)
            logger.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def unzip_file(self, zip_data_filepath: str, unzip_dir_path: str) -> Path:
        try:
            logger.info("Entered the unzip_file method of Data ingestion class")
            with ZipFile(zip_data_filepath, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir_path)
            logger.info("Exited the unzip_file method of Data ingestion class")
            
            return unzip_dir_path

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e 
        

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logger.info("Entered the initiate_data_ingestion method of Data ingestion class")

            # Creating Data Ingestion Artifacts directory inside artifact folder
            os.makedirs(self.data_ingestion_config.data_ingestion_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_ingestion_config.data_ingestion_artifacts_dir)} directory."
            )
            self.get_data_from_s3(bucket_file_name=S3_TRAIN_DATA_FILE_NAME, bucket_name=BUCKET_NAME, output_filepath=self.data_ingestion_config.train_data_download_path)
            logger.info(f"Downloaded train data from s3 bucket. File name - {os.path.basename(self.data_ingestion_config.train_data_download_path)}")

            self.get_data_from_s3(bucket_file_name=S3_EVAL_DATA_FILE_NAME, bucket_name=BUCKET_NAME, output_filepath=self.data_ingestion_config.eval_data_download_path)
            logger.info(f"Downloaded eval data from s3 bucket. File name - {os.path.basename(self.data_ingestion_config.eval_data_download_path)}")

            self.unzip_file(zip_data_filepath=self.data_ingestion_config.train_data_download_path, unzip_dir_path=self.data_ingestion_config.unzip_data_folder_path)
            logger.info("Extracted data from zipped file")

            self.unzip_file(zip_data_filepath=self.data_ingestion_config.eval_data_download_path, unzip_dir_path=self.data_ingestion_config.unzip_data_folder_path)
            logger.info("Extracted data from zipped file")

            dataset_path = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, TRAIN_DATA_UNZIP_FOLDER_NAME)
            labels = [label for label in os.listdir(dataset_path)]
            idx2label = {v: k for v, k in enumerate(labels)}
            label2idx = {k: v for v, k in enumerate(labels)}

            self.utils.dump_json_file(data=idx2label, filepath=self.data_ingestion_config.idx2label_file_path)
            self.utils.dump_json_file(data=label2idx, filepath=self.data_ingestion_config.label2idx_file_path)
            logger.info("Dumped index to label and label to index as json files")

            data_ingestion_artifacts = DataIngestionArtifacts(train_data_path=self.data_ingestion_config.train_data_download_path,
                                                              eval_data_path=self.data_ingestion_config.eval_data_download_path,
                                                              idx2label_file_path=self.data_ingestion_config.idx2label_file_path,
                                                              label2idx_file_path=self.data_ingestion_config.label2idx_file_path)                                 
            logger.info("Exited the initiate_data_ingestion method of Data ingestion class")

            return data_ingestion_artifacts
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        