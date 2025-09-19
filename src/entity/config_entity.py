import os
from src import constants
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=constants.PIPELINE_NAME
        self.artifact_name=constants.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp:str=timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, 
            constants.DATA_INGESTION_FEATURE_STORE_DIR, 
            constants.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, 
            constants.DATA_INGESTION_INGESTED_DIR, 
            constants.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, 
            constants.DATA_INGESTION_INGESTED_DIR, 
            constants.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = constants.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = constants.DATA_INGESTION_DATABASE_NAME