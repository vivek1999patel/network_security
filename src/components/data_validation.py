import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from src.logging.logger import logging
from src.constants import SCHEMA_FILE_PATH
from src.entity.config_entity import DataValidationConfig
from src.utils.utils import read_yaml_file, write_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.exception import NetworkSecurityException
from src.entity.artifact_entity import DataValidationArtifact

class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_num_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            num_of_schema_columns=len(self._schema_config)
            num_of_dataframe_columns=len(dataframe.columns)
            logging.info(f"Required number of columns: {num_of_schema_columns}")
            logging.info(f"Number of columns in Dataframe: {num_of_dataframe_columns}")
            if num_of_dataframe_columns == num_of_schema_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_data_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status":is_found
                    }
                })
            drift_report_file_path=self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            # Read data from train and test files
            train_df=DataValidation.read_data(train_file_path)
            test_df=DataValidation.read_data(test_file_path)

            # Validate num of columns
            status=self.validate_num_of_columns(train_df)
            if not status:
                error_message=f"Train data does not contain all column. \n"
            status=self.validate_num_of_columns(test_df)
            if not status:
                error_message=f"Test data does not contain all column. \n"

            # Check Data drift and save data under validated dir
            status=self.detect_data_drift(base_df=train_df, current_df=test_df)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)