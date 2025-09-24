import os
import sys
from src.logging.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.exception.exception import NetworkSecurityException
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact 
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
    
    def start_data_ingestion(self):
        try:
            data_ingestion_config=DataIngestionConfig(self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed and Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("Initiating Data Validation")
            data_validation_config=DataValidationConfig(self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact, data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Initiating Data Transformation")
            data_transformation_config=DataTransformationConfig(self.training_pipeline_config)
            data_transformation=DataTransformation(data_validation_artifact, data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed and Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Initiating Model Training")
            model_trainer_config=ModelTrainerConfig(self.training_pipeline_config)
            model_trainer=ModelTrainer(data_transformation_artifact, model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info(f"Model Training completed and Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)