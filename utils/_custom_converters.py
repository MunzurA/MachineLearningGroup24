import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

#------------------------------------------------------------------------------

class PercentageConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the percentage features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Convert the percentage string to a numerical value.
        """
        return X.str.rstrip('%').astype(float) / 100
    

class ListConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the list features, either with low (one-hot encoding) or high (targtet encoding) cardinality.

    Parameters:
        card_type (str): The cardinality type of the list feature, either 'low' or 'high'.
    """

    def __init__(self, card_type: str):
        if card_type not in ['low', 'high']:
            raise ValueError("card_type must be either 'low' or 'high'")
        
        self.card_type = card_type
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.target_mean = None

                 
    def fit(self, X, y=None):
        """
        Fit the encoder based ont he cardinality type.
        For 'low', fit OneHotEncoder, for 'high', compute target encoding.
        """
        X_cleaned = self._clean(X)

        if self.card_type == 'low':
            all_amenities = [item for sublist in X_cleaned for item in sublist]
            self.onehot_encoder.fit(np.array(all_amenities).reshape(-1, 1))
        elif self.card_type == 'high' and y is not None:
            target_encoded = self._target_encode(X_cleaned, y)
            self.target_mean = target_encoded.groupby('amenity')['price'].mean()
        
        return self
    
    def transform(self, X):
        """
        Transform the list features either using OneHotEncoder or target encoding base on the card_type parameter.
        """
        X_cleaned = self._clean(X)

        if self.card_type == 'low':
            flat_list = [item for sublist in X_cleaned for item in sublist]
            encoded = self.onehot_encoder.transform(np.array(flat_list).reshape(-1, 1))
            return pd.DataFrame(encoded, index=X.index).groupby(X.index).max()
        
        elif self.card_type == 'high' and self.target_mean is not None:
            encoded_amenities = X_cleaned.apply(lambda x: [self.target_mean.get(i, 0) for i in x])
            return pd.DataFrame(encoded_amenities.tolist(), index=X.index)

    def _clean(self, X) -> list:
        """
        Clean the list string.

        Returns:
            list: The cleaned list.
        """
        return X.apply(lambda x: [i.lower() for i in eval(x)])
    
    def _target_encode(self, X, y):
        """
        
        """
        amenities_flat = [item for sublist in X for item in sublist]
        amenities_df = pd.DataFrame(amenities_flat, columns=['amenity'])
        amenities_df['price'] = y.repeat(X.apply(len)).reset_index(drop=True)
        return amenities_df

    
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
        return self

    def transform(self, X):
        """
        Convert the boolean string to a boolean value (1/0).
        """
        return X.map(self.map)
    

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
        return self
    
    def transform(self, X):
        """
        Convert the date string to the difference in days from the baseline date.
        """
        return (self.baseline_date - datetime.strptime(X, self.format)).days
    