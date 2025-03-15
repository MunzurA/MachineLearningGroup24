import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from category_encoders import TargetEncoder

#------------------------------------------------------------------------------

class MultiLabelOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder for the low cardinality list features, using multi-label binarization.
    """
    def fit(self, X, y=None):
        self.encoders = {col: MultiLabelBinarizer() for col in X.columns}
        for col in X.columns:
            self.encoders[col].fit(X[col])
        return self
    
    def transform(self, X):
        """
        Convert the list(s) to a one-hot-encoded matrix.
        """
        result = []
        for col in X.columns:
            encoded = self.encoders[col].transform(X[col])
            result.append(pd.DataFrame(encoded, columns=self.encoders[col].classes_))
        return pd.concat(result, axis=1)
    

class MultiLabelTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder for the high cardinality list features, using multi-label target encoding.
    """
    def __init__(self):
        self.encoders = TargetEncoder()


    def fit(self, X, y=None):
        """
        Fit the target encoder to the exploded data.
        """
        exploded = X.explode()
        self.encoder.fit(exploded, y.loc[exploded.index])
        return self

    def transform(self, X):
        """
        Convert the list(s) to a one-hot-encoded matrix.
        """
        exploded = X.explode()
        encoded = self.encoder.transform(exploded)
        return encoded.groupoby(encoded.index).mean()