import os
import sys
import mlflow
import numpy as np
from src.logging.logger import logging
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    RandomForestClassifier
)
from src.entity.config_entity import ModelTrainerConfig
from src.utils.ml_utils.model.estimator import NetworkModel
from src.exception.exception import NetworkSecurityException
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric, feature, label):
        try:
            with mlflow.start_run():
                f1_score=classification_metric.f1_score
                precision_score=classification_metric.precision_score
                recall_score=classification_metric.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)

                signature=infer_signature(feature, label)
                mlflow.sklearn.log_model(best_model, "model", signature=signature)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }  
            }

            model_report:dict=evaluate_models(x_train, y_train, x_test, y_test, models, params)

            # Get best model score and best model name
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            y_train_pred=best_model.predict(x_train)
            classification_train_metric=get_classification_score(y_train, y_train_pred)
            
            # Track the experiments with mlflow for training data
            self.track_mlflow(best_model, classification_train_metric, x_train, y_train)

            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_test, y_test_pred)

            # Track the experiments with mlflow for test data
            self.track_mlflow(best_model, classification_test_metric, x_test, y_test)

            preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model=NetworkModel(preprocessor, best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)

            # Create model trainer artifact
            model_trainer_artifact=ModelTrainerArtifact(
                trainer_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testing array
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            # Feature-label split
            x_train, y_train, x_test, y_test=(
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Train model
            model_trainer_artifact=self.train_model(x_train, y_train, x_test, y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)