import os
import sys

from src.logger import logging
from src.exception import CustomException


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the data injestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Datadet read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path, index=False)
            logging.info('Raw Data is created')

            train_set, test_set=train_test_split(df,test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestionconfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path, index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )




        except Exception as e:
            logging.info('Exception occured at Data ingestion Stage')
            raise CustomException(e, sys)
            