import os 
import sys 
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path =  os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test,y_test = (
                train_arr[:, 1:-1],
                train_arr[:, 0],
                test_arr[:, 1:-1],
                test_arr[:, 0]
            )

            X_train = X_train[:-1]
            X_test = X_train[:-1]

            models = {
                'Decision Tree Regression' : DecisionTreeRegressor(),
                'Random Forest Regression' : RandomForestRegressor(),
                'AdaBoost Regression' : AdaBoostRegressor(),
                'Linear Regression' : LinearRegression(),
                'Gradient Boosting Regression' : GradientBoostingRegressor(),
                'Cat Boost Regression' : CatBoostRegressor(),
                'XGB Regression' : XGBRegressor(),
                'K Nearest Regressor' : KNeighborsRegressor()
            }

            params={
                "Decision Tree Regression": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest Regression":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regression":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},

                "XGB Regression":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boost Regression":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regression":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K Nearest Regressor":{
                    'n_neighbors':[5, 10, 20],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                }
                
            }

            logging.info(type(models))
            model_report:dict = evaluate_model(X_train=X_train, 
                                               y_train=y_train, 
                                               X_test=X_test, 
                                               y_test=y_test, 
                                               models=models, 
                                               param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score <= 0.5:
                raise CustomException('No best model found')
            logging.info('Best found model on both training and testing dataset')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return f'Best model : {best_model}, {r2_square}\nRest models : {model_report}'

        except Exception as e:
            raise CustomException(e, sys)
    