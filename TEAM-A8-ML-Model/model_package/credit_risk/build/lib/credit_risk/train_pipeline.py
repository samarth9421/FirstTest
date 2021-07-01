# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:32:40 2021

@author: Team A8
"""

# from sklearn.model_selection import train_test_split
# import pandas as pd
from credit_risk.pipeline import risk_pipe
from credit_risk.config import config
from credit_risk.data_processing.data_management import load_dataset, save_pipeline, save_lime_explainer

from lime import lime_tabular
  

def train():
    """Train the model
        Save lime explainer
    """

    data = load_dataset(file_name=config.TRAIN_DATA_FILE)

    X_train = data[config.FEATURES]
    y_train = data[config.TARGET].squeeze()
    
    risk_pipe.fit(X_train, y_train)
    
    pp_data = risk_pipe.only_transform(X_train)
    final_features = risk_pipe['onehot_encoder'].final_feature_names
    
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=pp_data,
        feature_names=final_features,
        class_names=['will repay', 'will default'],
        mode='classification',
        verbose=1
        )

    save_lime_explainer(explainer=explainer)
    save_pipeline(pipe=risk_pipe)


if __name__ == "__main__":
    train()
