import os
import sys
import logging
import torch
import warnings
from tqdm.auto import tqdm
from transformers import LayoutLMv2ForSequenceClassification, AdamW
from document_classifier.exception import DocumentClassifierException
from document_classifier.utils.main_utils import MainUtils
from document_classifier.constant import *
from document_classifier.entity.config_entity import ModelTrainerConfig
from document_classifier.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, DataIngestionArtifacts

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifacts: DataTransformationArtifacts,
                 data_ingestion_artifacts: DataIngestionArtifacts) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logger.info("Entered the initiate_model_trainer method of Model trainer class")

            os.makedirs(self.model_trainer_config.model_trainer_artifacts_dir, exist_ok=True)
            logger.info(f"Created {os.path.basename(self.model_trainer_config.model_trainer_artifacts_dir)} directory.")

            label2idx = self.utils.read_json_file(filepath=self.data_ingestion_artifacts.label2idx_file_path)
            logger.info("Loaded label to idex file from artifacts")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",  num_labels=len(label2idx))
            model.to(device)
            logger.info("Loaded pretrained LayoutLMV model")

            train_dataloader = self.utils.load_object(file_path=self.data_transformation_artifacts.train_dataloader_filepath)
            test_dataloader = self.utils.load_object(file_path=self.data_transformation_artifacts.test_dataloader_filepath)
            logger.info("Loaded train and test dataloader")

            print("==================== Training Started ======================")
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            num_epochs = EPOCHS

            for epoch in range(num_epochs):
                print("Epoch:", epoch)
                training_loss = 0.0
                training_correct = 0
                #put the model in training mode
                model.train()
                for batch in tqdm(train_dataloader):
                    outputs = model(**batch)
                    loss = outputs.loss

                    training_loss += loss.item()
                    predictions = outputs.logits.argmax(-1)
                    training_correct += (predictions == batch['labels']).float().sum()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                print("Training Loss:", training_loss / batch["input_ids"].shape[0])
                training_accuracy = 100 * training_correct / self.data_transformation_artifacts.train_data_len
                print("Training accuracy:", training_accuracy.item())  

                model.eval()
                validation_loss = 0.0
                validation_correct = 0
                
                for batch in tqdm(test_dataloader):
                    outputs = model(**batch)
                    loss = outputs.loss

                    validation_loss += loss.item()
                    predictions = outputs.logits.argmax(-1)
                    validation_correct += (predictions == batch['labels']).float().sum()

                print("Validation Loss:", validation_loss / batch["input_ids"].shape[0])
                validation_accuracy = 100 * validation_correct / self.data_transformation_artifacts.test_data_len
                print("Validation accuracy:", validation_accuracy.item())

            model.save_pretrained(self.model_trainer_config.trained_model_path)
            logger.info("Saved model to artifacts directory")

            saved_model_path = os.path.join(ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, SAVED_MODEL_DIR)

            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=saved_model_path)
            
            logger.info("Exited the initiate_model_trainer method of Model trainer class")
            return model_trainer_artifacts

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
