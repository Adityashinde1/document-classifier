import sys
import logging
from document_classifier.cloud_storage.s3_operations import S3Operation
from document_classifier.constant import *
from document_classifier.entity.artifacts_entity import ModelEvaluationArtifacts, ModelPusherArtifacts
from document_classifier.entity.config_entity import ModelPusherConfig
from document_classifier.exception import DocumentClassifierException

logger = logging.getLogger(__name__)

class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifacts,
        model_pusher_config: ModelPusherConfig,
    ) -> None:
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.s3_operation = S3Operation()


    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        try:
            logger.info("Enetred the initiate_model_pusher method of Model pusher class")
            if self.model_evaluation_artifact.is_model_accepted == True:
                
                accepted_model_folder = os.path.dirname(self.model_pusher_config.s3_model_path)

                # Uploading the model to s3 bucket
                self.s3_operation.sync_folder_to_s3(folder=accepted_model_folder, bucket_name=BUCKET_NAME, bucket_folder_name=SAVED_MODEL_DIR)
                logger.info("Model pushed to S3 bucket")
            
            else:
                raise Exception("Trained model is not better than the S3 best model")

            model_pusher_artifacts = ModelPusherArtifacts(
                bucket_name=BUCKET_NAME,
                trained_model_path=self.model_pusher_config.s3_model_path,
            )

            logger.info(
                "Exited the initiate_model_pusher method of Model pusher class"
            )
            return model_pusher_artifacts

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e