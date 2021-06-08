# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 00:45:17 2021

@author: Team A8
"""
from credit_risk.config import config
from credit_risk.data_processing import preprocessors as pp
from credit_risk.data_processing.myPipeline import PipelineCustom as Pipeline

# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler#, OneHotEncoder

import joblib

# onehot = ColumnTransformer(transformers=[('onehot',
#                                           OneHotEncoder(sparse = False,
#                                                         drop ='first'),
#                                           config.CATEGORICAL_OH,
#                                           )
#                                          ],
#                            remainder ='passthrough'
#                            )

model_dir = config.UNTRAINED_MODEL_DIR / "untrained_model.pkl"

with open(model_dir, 'rb') as input:
    model = joblib.load(input)

pipe_list=[
    ("numerical_imputer", pp.NumericalImputer(features=config.NUMERICAL)),
             
    ("categorical_encoder", pp.CategoricalEncoder(features=config.CATEGORICAL_LE)),
                
    ("onehot_encoder", pp.EncoderOneHot(oh_features=config.CATEGORICAL_OH,
                                        features_not_oh=config.FEATURES_NOT_OH
                                        )),
    
    ("standard_scaler", StandardScaler()),
    
    ('model',model)
   ]

risk_pipe = Pipeline(pipe_list)
