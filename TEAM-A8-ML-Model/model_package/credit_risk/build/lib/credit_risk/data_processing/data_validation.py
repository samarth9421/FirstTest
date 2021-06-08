# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 04:35:05 2021

@author: Team A8
"""

from credit_risk.config import config
from credit_risk.data_processing.errors import InvalidInputError

import pandas as pd


def validate_data(input_data):
    """Check model inputs for unexpected values.
    (This function is not that important right now. Including it for later)
    
    Parameters
    -----------
    input_data: pd.DataFrame
        Input data to be validated
    
    Returns
    --------
        Validated data
    
    """

    validated_data = input_data.copy()


    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NO_NA].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )
        # raise InvalidInputError( "Not allowed NaN values in categorical columns")

    return validated_data