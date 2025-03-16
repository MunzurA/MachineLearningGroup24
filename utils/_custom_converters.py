import numpy as np
import pandas as pd

from datetime import datetime
from ast import literal_eval

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

#------------------------------------------------------------------------------

class PercentageConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the percentage features.
    """
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        Convert the percentage string to a numerical value by removing the '%' sign and dividing by 100.
        """
        def func(X, y=None):
            X = pd.Series(X).copy()
            X = X.str.replace('%', '').astype(float) / 100
            return X
        
        return X.apply(func)
    
    def get_feature_names_out(self, input_features=None):
        """
        Return the feature names for output features.
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Transformer has not been fitted yet.")
        return input_features
    

class ListConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the list features.
    """
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        """
        Convert the list string literal to a list of strings.
        """
        def func(X, y=None):
            X = pd.Series(X).copy()
            X = X.apply(lambda x: literal_eval(x) if isinstance(x, str) else '[]')
            return X

        return X.apply(func)
    
    def get_feature_names_out(self, input_features=None):
        """
        Return the feature names for output features.
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Transformer has not been fitted yet.")
        return input_features

    
class BooleanConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the boolean features.

    Parameters:
        true_val (str): The value representing True. Defaults to 't'.
        false_val (str): The value representing False. Defaults to 'f'.
    """
    def __init__(self, true_val: str = 't', false_val: str = 'f'):
        self.true_val = true_val
        self.false_val = false_val
        self.map = {self.true_val: 1, self.false_val: 0}

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        Convert the boolean string to a boolean value (1/0).
        """
        def func(X, y=None):
            X = pd.Series(X).copy()
            X = X.map(self.map)
            return X
        
        return X.apply(func)
    
    def get_feature_names_out(self, input_features=None):
        """
        Return the feature names for output features.
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Transformer has not been fitted yet.")
        return input_features
    

class DateConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the date features.

    Parameters:
        baseline_date (datetime): The baseline date to calculate the difference from. Defaults to 2025-03-01.
        format (str): The format of the date string. Defaults to '%Y-%m-%d'.
    """
    def __init__(self, baseline_date: datetime = datetime(2025, 3, 1), format: str = '%Y-%m-%d'):
        self.baseline_date = baseline_date
        self.format = format

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        """
        Convert the date string to the difference in days from the baseline date.
        """
        def func(X, y=None):
            X = pd.Series(X).copy()
            X = pd.to_datetime(X)
            X = (self.baseline_date - X).dt.days
            return X
            
        return X.apply(func)
    
    def get_feature_names_out(self, input_features=None):
        """
        Return the feature names for output features.
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Transformer has not been fitted yet.")
        return input_features
    