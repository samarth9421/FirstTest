# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 02:35:22 2021

@author: Team A8
"""

import pandas as pd

from credit_risk.data_processing.data_management import load_pipeline, load_lime_explainer, save_figure
from credit_risk.config import config
from credit_risk.data_processing.data_validation import validate_data

#load trained pipeline
risk_pipe = load_pipeline(file_name=config.TRAINED_MODEL_FILE)

#load lime explainer object
explainer = load_lime_explainer(file_name=config.LIME_EXPLAINER_FILE)



def predict(input_data):
    """
    Make a prediction using the saved model pipeline.
    
    Parameters
    -----------
    input_data: JSON data
        Raw input data. Can contain any number of rows.
        
    Returns
    --------
    prediction: array of shape(Num_samples,)
        Predictions for each sample
    
    """

    data = pd.read_json(input_data)
    valid_data = validate_data(data)
    prediction = risk_pipe.predict_proba(valid_data[config.FEATURES])[:,1]

    return prediction

def predict_single(input_data, applicationID):
    """
    Predicts risk for a given applicationID
    
    Parameters
    -----------
    input_data: JSON data
        Database of all customers
    
    applicationID: int
        applicationID of the customer
        
    Returns
    --------
    prediction: array of shape(1,)
    """
    
    data = pd.read_json(input_data)
    data_row = data.loc[data['SK_ID_CURR'] == applicationID]
    
    valid_data = validate_data(data_row)
    prediction = risk_pipe.predict_proba(valid_data[config.FEATURES])[:,1]
    
    return prediction
    

def individual_interpretation(input_data, file_path, applicationID=None):
    """
    Saves lime interpretation figure
    
    For a single row no need to enter application id.
    
    Specify application if input_data contains multiple samples
    
    Parameters
    -----------
    input_data: JSON data
        Raw input data
    
    file_path: str
        Path to where the figure needs to be saved
        
    applicationID: int (default:None)
        applicationID of the customer. If None, only 1 row expected 
        in input data
        
    Returns
    --------
    
        
    
    """
    
    data = pd.read_json(input_data)
    
    if(applicationID == None and data.shape[0] != 1):
        return -1
    
    elif applicationID == None:
        data_row = data
    
    else:
        data_row = data.loc[data['SK_ID_CURR'] == applicationID]
        
    if len(data_row) == 0:
        return -1
        
    data_row = data_row[config.FEATURES]
    valid_data = validate_data(data_row)
    pp_data = risk_pipe.only_transform(valid_data)

    exp = explainer.explain_instance(
        data_row=pp_data[0],
        predict_fn=risk_pipe['model'].predict_proba
        )
    fig = exp.as_pyplot_figure()
    save_figure(fig, file_path)
    return 1


