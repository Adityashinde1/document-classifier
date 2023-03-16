import io
import logging
import os
import sys

import torch
from PIL import Image
from transformers import (LayoutLMv2FeatureExtractor,
                          LayoutLMv2ForSequenceClassification,
                          LayoutLMv2Processor, LayoutLMv2Tokenizer)

from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.constant import *
from document_classifier.exception import DocumentClassifierException
from document_classifier.utils.main_utils import MainUtils

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self) -> None:
        self.utils = MainUtils()
        self.feature_extractor = LayoutLMv2FeatureExtractor()
        self.tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.processor = LayoutLMv2Processor(self.feature_extractor, self.tokenizer)
        self.s3_operation = S3Operation()


    def doc_prediction(self, image_: Image,device: str, model: object, label2idx: dict) -> str:
        try:
            logger.info("Entered the doc_prediction method of Model predictor class")

            encoded_inputs = self.processor(image_, return_tensors="pt").to(device)
            logger.info("encoded inputs")
            outputs = model(**encoded_inputs)
            logger.info("got outputs")
            preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
            logger.info("got preds")
            pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
            label = max(pred_labels)
            logger.info("got pred labels")
            logger.info("Exited the doc_prediction method of Model predictor class")
            return label

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def initiate_model_predictor(self, image_bytes: bytes) -> dict:
        try:
            logger.info("Entered the initiate_model_predictor method of Model predictor class")

            # Convert bytes to image
            orig = Image.new(mode="RGB", size=(754, 1000))
            stream = io.BytesIO(image_bytes)
            orig.save(stream, 'PNG')
            image = Image.open(stream)  
            logger.info("Image bytes converted into image")

            self.s3_operation.sync_folder_from_s3(folder=BEST_MODEL_DIR, bucket_name=BUCKET_NAME, bucket_folder_name=SAVED_MODEL_DIR)
            logger.info("Best model downloaded from s3 bucket for prediction")

            self.s3_operation.download_file(bucket_name=BUCKET_NAME, output_file_path=os.path.join(LABEL_TO_IDX_FILE_NAME), key=LABEL_TO_IDX_FILE_NAME)
            label2idx_path = os.path.join(LABEL_TO_IDX_FILE_NAME)
            label2idx = self.utils.read_json_file(filepath=label2idx_path)
            logger.info("Label to index file loaded")

            model = LayoutLMv2ForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
            model.to("cpu")

            label = self.doc_prediction(image_=image,device="cpu", model=model, label2idx=label2idx)

            print(label)
            logger.info("Exited the initiate_model_predictor method of Model predictor class")
            
            return label

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        