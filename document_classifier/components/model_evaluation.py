import os
import sys
import logging
import torch
from tqdm.auto import tqdm
from transformers import LayoutLMv2ForSequenceClassification
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.entity.config_entity import ModelEvaluationConfig
from document_classifier.utils.main_utils import MainUtils
from document_classifier.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts
from document_classifier.exception import DocumentClassifierException
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor
from document_classifier.constant import *

logger = logging.getLogger(__name__)

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, 
                 data_transformation_artifacts: DataTransformationArtifacts) -> None:
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.utils = MainUtils()
        self.s3_operation = S3Operation()
        self.feature_extractor = LayoutLMv2FeatureExtractor()
        self.tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.processor = LayoutLMv2Processor(self.feature_extractor, self.tokenizer)


    def model_evaluate(self, model: object, device: str, eval_dataloader:object):
        try:
            logger.info("Entered the model_evaluate method of Model evaluation class")
            model.to(device)

            validation_loss = 0.0
            validation_correct = 0

            for batch in tqdm(eval_dataloader):
                outputs = model(**batch)
                loss = outputs.loss

                validation_loss += loss.item()
                predictions = outputs.logits.argmax(-1)
                validation_correct += (predictions == batch['labels']).float().sum()

            validation_accuracy = 100 * validation_correct / self.data_transformation_artifacts.eval_data_len
            logger.info("Exited the model_evaluate method of Model evaluation class")

            return validation_accuracy.item()

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logger.info("Entered the initiate_model_evaluation method of Model evaluation class")

            # Creating Model evaluation Artifacts directory inside artifacts folder
            os.makedirs(
                self.model_evaluation_config.model_evaluation_artifacts_dir,
                exist_ok=True,
            )
            logger.info(
                f"Created {os.path.basename(self.model_evaluation_config.model_evaluation_artifacts_dir)} directory."
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = LayoutLMv2ForSequenceClassification.from_pretrained(self.model_evaluation_config.trained_model_path)
            model.to(device)

            eval_dataloader = self.utils.load_object(file_path=self.data_transformation_artifacts.eval_dataloader_filepath)
            logger.info("Loaded evaluation dataloader from the artifacts folder")

            trained_model_accuracy = self.model_evaluate(model=model, device=device, eval_dataloader=eval_dataloader)
            logger.info(f"The trained model accuracy on eval dataset is - {trained_model_accuracy}")
            
            # Loading model from S3 bucket
            self.s3_operation.sync_folder_from_s3(folder=SAVED_MODEL_DIR, bucket_name=BUCKET_NAME, bucket_folder_name=SAVED_MODEL_DIR)

            # Checking whether S3 model folder exists in the root directory or not
            if not any(os.scandir(self.model_evaluation_config.s3_local_model_path)):
                tmp_best_model_score = 0
                logger.info("S3 model is not available locally for comparison.")
                

            else:
                logger.info("S3 model file available in the root directory")

                s3_model = torch.load(self.model_evaluation_config.s3_local_model_path)
                logger.info("S3 model loaded")

                s3_model_accuracy = self.model_evaluate(model=s3_model, device=device, eval_dataloader=eval_dataloader)
                logger.info(
                    f"Calculated the s3 model's Test accuracy. - {s3_model_accuracy}"
                )
                tmp_best_model_score = s3_model_accuracy

            is_model_accepted = trained_model_accuracy > tmp_best_model_score

            if is_model_accepted is True:
                accepted_model_path: str = self.model_trainer_artifacts.trained_model_path
            else:
                accepted_model_path = None

            model_evaluation_artifact = ModelEvaluationArtifacts(
                trained_model_accuracy=trained_model_accuracy,
                is_model_accepted=is_model_accepted,
                accepted_model_path=accepted_model_path
            )
            logger.info("Exited the initiate_model_evaluation method of Model evaluation class")
            return model_evaluation_artifact
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e