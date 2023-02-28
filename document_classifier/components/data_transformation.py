import os
import sys
import logging
import torch
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from document_classifier.utils.main_utils import MainUtils
from document_classifier.exception import DocumentClassifierException
from document_classifier.entity.config_entity import DataTransformationConfig
from document_classifier.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
from document_classifier.constant import *
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D

logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts) -> None:
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.utils = MainUtils()
        self.feature_extractor = LayoutLMv2FeatureExtractor()
        self.tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.processor = LayoutLMv2Processor(self.feature_extractor, self.tokenizer)


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

            images = []
            labels = []

            for label in os.listdir(data_path):
                images.extend([f"{data_path}/{label}/{img_name}" for img_name in os.listdir(f"{data_path}/{label}")])
                labels.extend([label for _ in range(len(os.listdir(f"{data_path}/{label}")))])

            data_frame = pd.DataFrame({'image_path': images, 'label': labels})
            logger.info("Exited the convert_data_to_dataframe method of Data ingestion class")

            return data_frame
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def encode_example(self, examples):
        try:
            label2idx = self.utils.read_json_file(filepath=self.data_ingestion_artifacts.label2idx_file_path)
            images = [Image.open(path).convert("RGB") for path in examples['image_path']]
            encoded_inputs = self.processor(images, padding="max_length", truncation=True)
            encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]

            return encoded_inputs    
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
                

    def dataloader_from_df(self, data,features, device, shuffle = False):
        try:
            logger.info("Entered the training_dataloader_from_df method of Data transformation class")
            dataset = Dataset.from_pandas(data)
            
            encoded_dataset = dataset.map(
                self.encode_example,remove_columns=dataset.column_names, features=features, 
                batched=True, batch_size=2
            )
            encoded_dataset.set_format(type='torch', device=device)
            dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4, shuffle=shuffle)
            batch = next(iter(dataloader))
            logger.info("Exited the training_dataloader_from_df method of Data transformation class")
            return dataloader
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def initiate_data_transformation(self) -> DataTransformationArtifacts: 
        try:
            logger.info("Entered the initiate_data_transformation method of Data transformation class")

            os.makedirs(self.data_transformation_config.data_transformation_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_transformation_config.data_transformation_artifacts_dir)} directory."
            )
            label2idx = self.utils.read_json_file(filepath=self.data_ingestion_artifacts.label2idx_file_path)
            logger.info("Loaded label2idx file")

            train_dataset_path = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, TRAIN_DATA_UNZIP_FOLDER_NAME)
            train_df = self.convert_data_to_dataframe(data_path=train_dataset_path)
            logger.info("Created DataFrame from the train data")

            valid_dataset_path = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, EVAL_DATA_UNZIP_FOLDER_NAME)
            eval_df = self.convert_data_to_dataframe(data_path=valid_dataset_path)
            logger.info("Created DataFrame from the eval data")

            train_data, test_data = self.split_data(data=train_df, test_size=TEST_SIZE)
            logger.info("Splitted data into train and test dataset")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            training_features = Features({
                                        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                                        'input_ids': Sequence(feature=Value(dtype='int64')),
                                        'attention_mask': Sequence(Value(dtype='int64')),
                                        'token_type_ids': Sequence(Value(dtype='int64')),
                                        'bbox': Array2D(dtype="int64", shape=(512, 4)),
                                        'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
                                        })
  
            train_dataloader = self.dataloader_from_df(data=train_data, features=training_features, device=device, shuffle=True)
            test_dataloader = self.dataloader_from_df(data=test_data, features=training_features, device=device)
            logger.info("Generated Train and Test dataloader")

            eval_dataloader = self.dataloader_from_df(data=eval_df, features=training_features, device=device)
            logger.info("Generated Valid dataloader")

            self.utils.save_object(file_path=self.data_transformation_config.train_dataloader_filepath, obj=train_dataloader)
            self.utils.save_object(file_path=self.data_transformation_config.test_dataloader_filepath, obj=test_dataloader)
            self.utils.save_object(file_path=self.data_transformation_config.eval_dataloader_filepath, obj=eval_dataloader)
            logger.info("Dumped train, test and eval dataloader object")

            data_transformation_artifacts = DataTransformationArtifacts(train_dataloader_filepath=self.data_transformation_config.train_dataloader_filepath,
                                                                        test_dataloader_filepath=self.data_transformation_config.test_dataloader_filepath,
                                                                        eval_dataloader_filepath=self.data_transformation_config.eval_dataloader_filepath,
                                                                        train_data_len=len(train_data),
                                                                        test_data_len=len(test_data),
                                                                        eval_data_len=len(eval_df))
            logger.info("Exited the initiate_data_transformation method of Data transformation class")

            return data_transformation_artifacts
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        