import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 


class PredictPipeline:
    def __init__(self):
        pass

    def predictor(self, features):
        try:    
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, second:int, third:int, fourth:int, fifth:int, gender:str):
        self.second = second
        self.third = third
        self.fourth = fourth
        self.fifth = fifth
        self.gender = gender

    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                'second':[self.second],
                'third':[self.third],
                'fourth':[self.fourth],
                'fifth':[self.fifth],
                'gender':[self.gender]
            }

            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)