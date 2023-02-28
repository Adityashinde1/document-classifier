import sys
import logging
from document_classifier.components.data_ingestion import DataIngestion
from document_classifier.components.data_transformation import DataTransformation
from document_classifier.components.model_evaluation import ModelEvaluation
from document_classifier.components.model_pusher import ModelPusher
from document_classifier.components.model_trainer import ModelTrainer
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.constant import *
from document_classifier.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    ModelPusherArtifacts,
)
from document_classifier.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)
from document_classifier.exception import DocumentClassifierException

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.s3_operation = S3Operation()

    # This method is used to start the data ingestion
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logger.info("Getting the data from S3 bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config,  s3_operation=self.s3_operation
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the data from Google cloud storage")
            logger.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    # This method is used to start the data validation
    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        logger.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifacts=data_ingestion_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logger.info("Performed the data transformation operation")
            logger.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    # This method is used to start the model traidocument_classifier
    def start_model_training(
        self, data_transformation_artifacts: DataTransformationArtifacts, data_ingestion_artifacts: DataIngestionArtifacts
    ) -> ModelTrainerArtifacts:
        logger.info("Entered the start_model_training method of Train pipeline class")
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifacts=data_transformation_artifacts,
                data_ingestion_artifacts=data_ingestion_artifacts
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logger.info("Performed the Model training operation")
            logger.info(
                "Exited the start_model_training method of Train pipeline class"
            )
            return model_trainer_artifact

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    # This method is used to start model evaluation
    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
    ) -> ModelEvaluationArtifacts:
        try:
            logger.info(
                "Entered the start_model_evaluation method of Train pipeline class"
            )
            model_evaluation = ModelEvaluation(
                data_transformation_artifacts=data_transformation_artifact,
                model_trainer_artifacts=model_trainer_artifacts,
                model_evaluation_config=self.model_evaluation_config,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logger.info(
                "Exited the start_model_evaluation method of Train pipeline class"
            )
            return model_evaluation_artifact

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    # This method is used to statr model pusher
    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifacts
    ) -> ModelPusherArtifacts:
        try:
            logger.info(
                "Entered the start_model_pusher method of Train pipeline class"
            )
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logger.info("Exited the start_model_pusher method of Train pipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    # This method is used to start the training pipeline
    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifacts=data_transformation_artifacts,
                data_ingestion_artifacts=data_ingestion_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifacts,
                model_trainer_artifacts=model_trainer_artifact,
            )
            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e