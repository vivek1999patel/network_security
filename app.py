import os
import sys
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv
from uvicorn import run as app_run
from src.logging.logger import logging
from fastapi.responses import Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from src.utils.main_utils.utils import load_object
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from src.constants import DATA_INGESTION_DATABASE_NAME
from src.constants import DATA_INGESTION_COLLECTION_NAME
from src.utils.ml_utils.model.estimator import NetworkModel
from src.pipeline.training_pipeline import TrainingPipeline
from src.exception.exception import NetworkSecurityException


ca=certifi.where()
load_dotenv()

MONGO_DB_URL=os.getenv('MONGO_DB_URL')
print(MONGO_DB_URL)

mongo_client=pymongo.MongoClient(MONGO_DB_URL, tlsCAfile=ca)
database=mongo_client[DATA_INGESTION_DATABASE_NAME]
collection=mongo_client[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()
origins=["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates=Jinja2Templates(directory="./templates")

@app.get("/", tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs") 

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("best_model/preprocessor.pkl")
        final_model=load_object("best_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
if __name__=="__main__":
    app_run(app, host="0.0.0.0", port=8080)