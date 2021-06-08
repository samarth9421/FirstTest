# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 04:13:34 2021

@author: Team A8
"""

import math

from credit_risk.inference import predict, individual_interpretation, predict_single
from credit_risk.data_processing.data_management import load_dataset
from credit_risk.config import config
from sklearn.metrics import roc_auc_score


def test_make_prediction():
    # Given
    test_data = load_dataset(file_name=config.TEST_DATA_FILE)
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = predict(input_data=single_test_json)
    # Then
    assert subject is not None
    assert math.ceil(subject) == 1
    
def test_save_lime_interpretation():
    #Given
    test_data = load_dataset(file_name=config.TEST_DATA_FILE)
    test_json = test_data.to_json(orient='records')
    applicationID = 249123
    save_path = 'figure'
    
    #When
    subject = individual_interpretation(input_data=test_json,
                                        applicationID=applicationID,
                                        file_path=save_path)
    
    #Then 
    assert subject == 1
    
def test_single_prediction():
    #Given
    test_data = load_dataset(file_name=config.TEST_DATA_FILE)
    test_json = test_data.to_json(orient='records')
    applicationID = 249123
    
    #When
    subject = predict_single(input_data=test_json, applicationID=applicationID)
    #Then
    assert subject is not None
    assert math.ceil(subject)== 1
    
    
def test_model_auc():
    
    #Given
    test_data = load_dataset(file_name=config.TEST_DATA_FILE)
    test_json = test_data.to_json(orient='records')
    y_test = test_data['TARGET']
    
    #When
    predictions = predict(input_data=test_json)
    subject = roc_auc_score(y_test, predictions)
    
    #Then
    assert subject is not None
    assert subject > 0.5
    assert math.ceil(subject) == 1
    
    
    
    
    
