import numpy as np 
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.utils import sav_object
from src.utils import evaluate_model
from src.logger import logging
from src.exception import CustomException

import os, sys
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and independent variables from train & test data')

            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Creating List of models
            models={
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'Elasticnet':ElasticNet(),
                'Tree':DecisionTreeRegressor(),
                'RandomF':RandomForestRegressor()  
            }
            #'LogisticRegression':LogisticRegression(),
            #'KNeighbor':KNeighborsRegressor()

            model_report:dict=evaluate_model(X_train,y_train,X_test, y_test,models)
            print('\n================================================================================\n')
            logging.info(f'Model Report :{model_report}')

            # To get best model score from dictionary

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            print(f'Best Model found, Model name: {best_model_name},R2_Score: {best_model_score}')
            print('\n==============================================================================\n')

            logging.info(f'Best Model found, Model name: {best_model_name},R2_Score: {best_model_score}')

            sav_object(file_path=self.model_trainer.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.info('Exception occured at model training')
            raise CustomException(e,sys)