# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:53:18 2021

@author: Team A8
"""

import pandas as pd
import joblib
import dill
from credit_risk.config import config
import mpld3
import os


def load_dataset(file_name:str):
    """
    Loads dataset from csv
    
    Parameters
    -----------
    file_name : str
        .csv file name of the dataset
           
    Returns
    --------
    data: pd.DataFrame
        Dataset as a Pandas Dataframe object   
    """
    try:
        data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
        
        assert isinstance(data, pd.core.frame.DataFrame), "data should be a pandas Dataframe"
        assert len(data) > 0, "dataset should not be null"

        return data
    except AssertionError as msg: 
        print(msg)
        return msg 


def save_pipeline(pipe):
    """
    Save the pipeline
    
    Parameters
    -----------
    pipe : Pipeline object
          
    Returns
    --------
    Saves pipeline in a predefined directory 
    """

    file_name = config.TRAINED_MODEL_FILE
    save_path = config.TRAINED_MODEL_DIR / file_name
    with open(save_path, 'wb') as output:
        joblib.dump(pipe, output)

    print("Pipeline Saved")


def load_pipeline(file_name: str):
    """
    Load a persisted pipeline
    
    Parameters
    -----------
    file_name : str
        file name of saved pipeline
              
    Returns
    --------
    saved pipeline: Pipeline object
        Prefitted pipeline
    """

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

def save_lime_explainer(explainer):
    """
    Save lime explainer object
    
    Parameters
    -----------
    explainer : Lime Explainer object      
    
    Returns
    --------
        Saves explainer in a predefined directory
    """
    
    file_name = config.LIME_EXPLAINER_FILE
    save_path = config.LIME_DIR / file_name
    with open(save_path, 'wb') as output:
        dill.dump(explainer, output)
    
    print("Lime Explainer Saved")

def load_lime_explainer(file_name):
    """
    Loads lime explainer object

    Parameters
    -----------
    file_name : str
        file name of the saved explainer object
        
    Returns
    --------
    explainer: Lime explainer object
        Trained lime explainer object      
    """
    
    file_path = config.LIME_DIR / file_name
    with open(file_path, 'rb') as inp:
        explainer = dill.load(inp)
    return explainer

def save_figure(fig, file_path:str):
    """
    Save figures
    
    Parameters
    -----------
    fig :
        Figure object to be saved
    
    file_path: str
        Path to where the figure should be saved
          
    Returns
    --------
        Saves figures in the specified location
    """

    try: 
        fig.savefig(file_path, transparent=False)
        
        assert os.path.exists(str(file_path)), "fig not saved"

        file_path_html = file_path + '.html'
        mpld3.save_html(fig, str(file_path_html), template_type='simple')

        assert os.path.exists(str(file_path_html)), "html file not saved"

        file_path_json = file_path+'.json'
        mpld3.save_json(fig, str(file_path_json))

        assert os.path.exists(str(file_path_json)), "json file not saved"

    except AssertionError as msg: 
        print(msg)
        return msg 
