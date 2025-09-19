import os
import sys
import pymongo
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv
from src.logging.logger import logging
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.exception import NetworkSecurityException

# Configuration of Data Ingetion Config
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
            Read data from MongoDB
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]
            df=pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"])

            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_data_into_train_test(self, dataframe: pd.DataFrame):
        try:
            train_test_split_ratio=self.data_ingestion_config.train_test_split_ratio
            train_df, test_df = train_test_split(dataframe, 
                                                 test_size=train_test_split_ratio, 
                                                 random_state=42)
            logging.info("Performed train test split on the dataframe")
            
            training_file_path=self.data_ingestion_config.training_file_path
            testing_file_path=self.data_ingestion_config.testing_file_path
            training_dir=os.path.dirname(training_file_path)
            os.makedirs(training_dir, exist_ok=True)

            logging.info("Exporting train and test data.")
            train_df.to_csv(training_file_path, index=False, header=True)
            test_df.to_csv(testing_file_path, index=False, header=True)
            logging.info("Exported train and test data.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_into_train_test(dataframe)
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)