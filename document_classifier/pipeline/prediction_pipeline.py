import os
import sys
import io
import logging
import torch
from PIL import Image
from document_classifier.constant import *
from document_classifier.utils.main_utils import MainUtils
from transformers import LayoutLMv2ForSequenceClassification
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.exception import DocumentClassifierException
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self) -> None:
        self.utils = MainUtils()
        self.feature_extractor = LayoutLMv2FeatureExtractor()
        self.tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.processor = LayoutLMv2Processor(self.feature_extractor, self.tokenizer)
        self.s3_operation = S3Operation()


    def doc_prediction(self, image_: Image, processor: object, device: str, model: object, label2idx: dict):
        try:
            logger.info("Entered the doc_prediction method of Model predictor class")
            encoded_inputs = processor(image_, return_tensors="pt").to(device)
            outputs = model(**encoded_inputs)
            preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
            pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}

            logger.info("Exited the doc_prediction method of Model predictor class")
            return pred_labels

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def initiate_model_predictor(self, image_bytes: bytes):
        try:
            logger.info("Entered the initiate_model_predictor method of Model predictor class")
            
            # Convert bytes to image
            orig = Image.new(mode="RGB", size=(754, 1000))
            stream = io.BytesIO(image_bytes)
            orig.save(stream, 'PNG')
            image = Image.open(stream)

            self.s3_operation.sync_folder_from_s3(folder=BEST_MODEL_DIR, bucket_name=BUCKET_NAME, bucket_folder_name=SAVED_MODEL_DIR)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.s3_operation.download_file(bucket_name=BUCKET_NAME, output_file_path=os.path.join(from_root(), LABEL_TO_IDX_FILE_NAME), key=LABEL_TO_IDX_FILE_NAME)
            label2idx_path = os.path.join(from_root(), LABEL_TO_IDX_FILE_NAME)
            label2idx = self.utils.read_json_file(filepath=label2idx_path)

            best_model_path = os.path.join(from_root(), BEST_MODEL_DIR)
            model = LayoutLMv2ForSequenceClassification.from_pretrained(best_model_path)
            model.to(device)

            label = self.doc_prediction(image_=image, processor=self.processor, device=device, model=model, label2idx=label2idx)
            print(label)
            return label

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e