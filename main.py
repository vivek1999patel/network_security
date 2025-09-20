import sys
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.exception.exception import NetworkSecurityException
from src.entity.config_entity import DataTransformationConfig
from src.components.data_transformation import DataTransformation

if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()

        logging.info("Initiating Data Ingestion")
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Initiation Completed")

        logging.info("Initiating Data Validation")
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Initiation completed")

        logging.info("Initiating Data Transformation")
        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Initiation Completed")

    except Exception as e:
        raise NetworkSecurityException(e, sys)