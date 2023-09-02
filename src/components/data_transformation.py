import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_columns = ['second','third','fourth','fifth']
            categorical_columns = ['gender']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    # ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('label_encoder', OneHotEncoder()),
                    # ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_columns}'),
            logging.info(f'Numerical Columns: {numeric_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_columns),
                    ('cat_pipelines', cat_pipeline, categorical_columns)
                ]
            )
            logging.info('Transforming Numeric and categorical Columns')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path) 

            logging.info('Reading train and test data completed')


            useful_colomns = ['first','second','third','fourth','fifth','gender']
            new_train_df = train_df[useful_colomns]
            new_test_df = test_df[useful_colomns]

            logging.info('Selecting useful columns')


            preprocessing_obj = self.get_data_transformer_object()

            logging.info('Obtaining preprocessing object')
            

            input_feature_train_df = new_train_df.iloc[:, 1:]
            target_feature_train_arr = new_train_df.iloc[:, 0].fillna((new_train_df.iloc[:, 0]).mean)


            input_feature_test_df = new_test_df.iloc[:, 1:]
            target_feature_test_arr = new_test_df.iloc[:, 0].fillna((new_test_df.iloc[:, 0]).mean)            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Splitting the train and test data into input and target sets')

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info('Applying preprocessing object on training and testing dataframe')


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Saved preprocessing object')

            
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
