import os
import sys
import json
import certifi
import pymongo
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException

load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

ca=certifi.where()

class MongoDBDataLoading():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def convert_csv_to_json(self, file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = data.to_dict(orient="records")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.records=records
            self.database=database
            self.collection=collection

            # Instantiate MongoClient to connect to database
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            
            # Insert data
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__=="__main__":
    FILE_PATH="data/phisingData.csv"
    DATABASE="NETWORKSECURITY"
    COLLECTION="PhisingData"
    data_loading_Obj=MongoDBDataLoading()
    records=data_loading_Obj.convert_csv_to_json(file_path=FILE_PATH)
    print(records)
    no_of_records_inserted=data_loading_Obj.insert_data_mongodb(records, DATABASE,COLLECTION)
    print(no_of_records_inserted)