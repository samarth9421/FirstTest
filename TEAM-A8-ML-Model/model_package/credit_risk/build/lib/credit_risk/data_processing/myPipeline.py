# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:33:30 2021

@author: utkar
"""

from sklearn.pipeline import Pipeline

class PipelineCustom(Pipeline):
    """
    Custom Pipline class inherited from sklearn.pipeline.Pipeline
    
    Methods
    --------
    only_transform: (X:pd.DataFrame)-> X: pd.DataFrame
    
    """
    
    def only_transform(self, X):
        """
        This method performs all the steps except the final step in
        the pipeline
        
        Parameters
        -----------
        X: pd.DataFrame
            Input Raw data
    
        Returns
        --------
        X: pd.DataFrame
            Preprocessed data
        """
            
        X = X.copy()
        
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
            
        return X

        