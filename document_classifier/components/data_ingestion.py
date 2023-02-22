import os
import sys
import logging
from pandas import DataFrame
import pandas as pd
from zipfile import ZipFile, Path
from document_classifier.utils.main_utils import MainUtils
from document_classifier.constant import *
from sklearn.model_selection import train_test_split
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
        

    def split_data(self, data: DataFrame, test_size: float) -> DataFrame:
        try:
            logger.info("Entered the split_data method of Data ingestion class")
            train_data, valid_data = train_test_split(data, test_size=test_size, stratify=data.label)
            train_data = train_data.reset_index(drop=True)
            valid_data = valid_data.reset_index(drop=True)
            logger.info("Exited the split_data method of Data ingestion class")

            return train_data, valid_data

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def convert_data_to_dataframe(self, data_path: str) -> DataFrame:
        try:
            logger.info("Entered the convert_data_to_dataframe method of Data ingestion class")
            labels = [label for label in os.listdir(data_path)]
            idx2label = {v: k for v, k in enumerate(labels)}
            label2idx = {k: v for v, k in enumerate(labels)}

            images = []
            labels = []

            for label in os.listdir(data_path):
                images.extend([f"{data_path}/{label}/{img_name}" for img_name in os.listdir(f"{data_path}/{label}")])
                labels.extend([label for _ in range(len(os.listdir(f"{data_path}/{label}")))])

            data_frame = pd.DataFrame({'image_path': images, 'label': labels})
            logger.info("Exited the convert_data_to_dataframe method of Data ingestion class")

            return idx2label, label2idx, data_frame
        
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
            self.get_data_from_s3(bucket_file_name=S3_DATA_FILE_NAME, bucket_name=BUCKET_NAME, output_filepath=self.data_ingestion_config.data_download_path)
            logger.info(f"Downloaded data from s3 bucket. File name - {os.path.basename(self.data_ingestion_config.data_download_path)}")

            self.unzip_file(zip_data_filepath=self.data_ingestion_config.data_download_path, unzip_dir_path=self.data_ingestion_config.unzip_data_folder_path)
            logger.info("Extracted data from from zipped file")

            idx2label, label2idx, data_frame = self.convert_data_to_dataframe(data_path=self.data_ingestion_config.unzip_data_folder_path)
            logger.info("Data converted into DataFrame")

            train_dataset, test_dataset = self.split_data(data=data_frame, test_size=TEST_SIZE)
            logger.info("Splitted data into train and validation set")

            self.utils.dump_pickle_file(output_filepath=self.data_ingestion_config.train_dataset_file_path, data=train_dataset)
            self.utils.dump_pickle_file(output_filepath=self.data_ingestion_config.test_dataset_file_path, data=test_dataset)
            logger.info("Dumped train and test dataset into pickle file")

            
            logger.info("Exited the initiate_data_ingestion method of Data ingestion class")
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e