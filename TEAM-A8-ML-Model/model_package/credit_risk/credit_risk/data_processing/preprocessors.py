# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:34:01 2021

@author: Team A8
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#No categorical data imputed in dataset so commenting CategoricalImputer

# class CategoricalImputer(BaseEstimator, TransformerMixin):
#     """Imputes Missing Categorical Data with Unknown"""
    
#     def __init__(self, features=None):
#         if not isinstance(features, list):
#             self.features = [features]
#         else:
#             self.features = features
            
#     def fit(self, X, y= None):
#         """No fitting needed"""
#         return self
    
#     def transform(self, X):
#         """Tranform missing values to unknown"""
#         X = X.copy()
#         for feature in self.features:
#             X[feature] = X[feature].fillna("Unknown")
            
#         return X

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Imputes missing numerical data with median value"""
    
    def __init__(self, features=None):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
            
    def fit(self, X: pd.DataFrame, y=None):
        """Fit every feature for its median value"""
        self.imputer_dict = {}
        
        for feature in self.features:
            self.imputer_dict[feature] = X[feature].median() #Maintains imputation values for each numerical feature
        
        return self
    
    
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        X = X.copy()
        for feature in self.features:
            X[feature].fillna(self.imputer_dict[feature], inplace=True)
        return X
            
            

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features"""
    
    def __init__(self, features=None):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X:pd.DataFrame, y=None):
        """Hardcoded for NAME_EDUCATION_TYPE feature"""
        
        self.encoder_dict = {"Lower secondary": 0,
                             "Secondary / secondary special":1,
                             "Incomplete higher" : 2,
                             "Higher education" : 3,
                             "Academic degree" : 4
                             }
        return self
    
    def transform(self, X:pd.DataFrame)-> pd.DataFrame:
        """Label encode NAME_EDUCATION_TYPE"""
        X = X.copy()
        for feature in self.features:
            for edu_type, edu_value in self.encoder_dict.items():
                X[feature].replace(to_replace=edu_type,
                                   value = edu_value,
                                   inplace=True
                                   )
        return X
    
    
class EncoderOneHot(BaseEstimator, TransformerMixin):
    """One hot encoder features"""
    
    def __init__(self, oh_features=None, features_not_oh=None):
        if not isinstance(oh_features, list):
            self.oh_features = [oh_features]
        else:
            self.oh_features = oh_features
            
        if not isinstance(features_not_oh, list):
            self.features_not_oh = [features_not_oh]
        else:
            self.features_not_oh = features_not_oh
            
            
    def fit(self, X:pd.DataFrame, y=None):
        Xt = X[self.oh_features]
        self.encoder = OneHotEncoder(sparse = False,drop ='first')
        self.encoder.fit(Xt)
        
        #To store final feature names after onehot encoding.
        #The dataframe is converted to numpy array afterwards so 
        #feature names are lost 
        self.final_feature_names = [] 
        self.final_feature_names.extend(self.features_not_oh)
        self.final_feature_names.extend(self.encoder.get_feature_names(self.oh_features))
        return self
        
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        X = X.copy()
        Xt = X[self.oh_features]
        transformed = self.encoder.transform(Xt)
        df_transformed = pd.DataFrame(transformed,
                                    columns = self.encoder.get_feature_names(self.oh_features)
                                    )
        
        X_tt = X[self.features_not_oh]   
        X_tt = X_tt.join(df_transformed)
        X_tt = X_tt[self.final_feature_names]  # Make sure the order of features in dataframe is correct for reproducibilty

        return X_tt
        
        
        
        
        
            
        
            
            
        
        
        
        
    
    
            