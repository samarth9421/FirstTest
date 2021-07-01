# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:50:21 2021

@author: Team A8
"""

import pathlib 
import credit_risk



ROOT_DIR = pathlib.Path(credit_risk.__file__).resolve().parent #Package path
SAVED_FILES_DIR = ROOT_DIR/"save_files"

TRAINED_MODEL_DIR = SAVED_FILES_DIR/"trained_models"
UNTRAINED_MODEL_DIR = SAVED_FILES_DIR/"untrained_models"
DATASET_DIR = SAVED_FILES_DIR/ "datasets"
LIME_DIR = SAVED_FILES_DIR/ "lime_estimator"
# LIME_PLOTS_DIR = SAVED_FILES_DIR/ "plots" / "lime"

LIME_EXPLAINER_FILE = "explainer.pkl"
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
TRAINED_MODEL_FILE = "trained_model.pkl"


#Target label
TARGET = ['TARGET']

#FEATURES TO BE SELECTED FROM RAW DATA
FEATURES = ['FLAG_DOCUMENT_8',
            'NAME_CONTRACT_TYPE',
            'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'NAME_EDUCATION_TYPE',
            'REGION_POPULATION_RELATIVE',
            'DAYS_BIRTH',
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
            'OWN_CAR_AGE',
            'FLAG_EMP_PHONE',
            'FLAG_WORK_PHONE',
            'FLAG_PHONE',
            'FLAG_EMAIL',
            'REGION_RATING_CLIENT',
            'LIVE_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'DEF_60_CNT_SOCIAL_CIRCLE',
            'DAYS_LAST_PHONE_CHANGE',
            ]

#ALL CATEGORICAL FEATURES
CATEGORICAL = ['NAME_CONTRACT_TYPE',
               'FLAG_EMP_PHONE',
               'REG_CITY_NOT_LIVE_CITY',
               'FLAG_OWN_CAR', 
               'FLAG_WORK_PHONE',
               'FLAG_EMAIL',
               'LIVE_REGION_NOT_WORK_REGION', 
               'LIVE_CITY_NOT_WORK_CITY',
               'FLAG_OWN_REALTY',
               'FLAG_PHONE', 
               'FLAG_DOCUMENT_8',
               'REG_CITY_NOT_WORK_CITY',
               'REGION_RATING_CLIENT',
               'NAME_EDUCATION_TYPE'
               ]

#ALL NUMERICAL FEATURES
NUMERICAL = [feature for feature in FEATURES if(feature not in CATEGORICAL) ]

#CATEGORICAL FEATURES TO BE LABEL ENCODER
CATEGORICAL_LE = ['NAME_EDUCATION_TYPE']

#CATEGORICAL FEATURES TO BE ONEHOT ENCODED
CATEGORICAL_OH = ['NAME_CONTRACT_TYPE',
                  'FLAG_EMP_PHONE',
                  'REG_CITY_NOT_LIVE_CITY',
                  'FLAG_OWN_CAR', 
                  'FLAG_WORK_PHONE',
                  'FLAG_EMAIL',
                  'LIVE_REGION_NOT_WORK_REGION', 
                  'LIVE_CITY_NOT_WORK_CITY',
                  'FLAG_OWN_REALTY',
                  'FLAG_PHONE', 
                  'FLAG_DOCUMENT_8',
                  'REG_CITY_NOT_WORK_CITY',
                  'REGION_RATING_CLIENT'
                  ]

#FEATURES WHICH ARE NOT ONEHOT ENCODED
FEATURES_NOT_OH = [feature for feature in FEATURES if(feature not in CATEGORICAL_OH)]


#FEATURES FOR WHICH NAN VALUES ARE NOT ALLOWED
CATEGORICAL_NO_NA = ['NAME_CONTRACT_TYPE',
               'FLAG_EMP_PHONE',
               'REG_CITY_NOT_LIVE_CITY',
               'FLAG_OWN_CAR', 
               'FLAG_WORK_PHONE',
               'FLAG_EMAIL',
               'LIVE_REGION_NOT_WORK_REGION', 
               'LIVE_CITY_NOT_WORK_CITY',
               'FLAG_OWN_REALTY',
               'FLAG_PHONE', 
               'FLAG_DOCUMENT_8',
               'REG_CITY_NOT_WORK_CITY',
               'REGION_RATING_CLIENT',
               'NAME_EDUCATION_TYPE'
               ]



