from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    train_data_path: str
    eval_data_path: str
    idx2label_file_path: str
    label2idx_file_path: str


# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    train_dataloader_filepath: str
    test_dataloader_filepath: str
    eval_dataloader_filepath: str
    train_data_len: int
    test_data_len: int
    eval_data_len: int


# Model trainer Artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str


# Model evaluation Artifacts
@dataclass
class ModelEvaluationArtifacts:
    trained_model_accuracy: float
    is_model_accepted: bool
    accepted_model_path: str


# Model pusher Artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str
    trained_model_path: str