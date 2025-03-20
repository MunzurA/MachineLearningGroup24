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
        self.feature_names_in_ = X.columns.tolist()

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
            result.append(pd.DataFrame(encoded, columns=self.encoders[col].classes_, index=X.index))

        return pd.concat(result, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        """
        Return the feature names for output features.
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Transformer has not been fitted yet.")
            
        # Generate the feature names from the classes combined with the original column names
        output_feature_names = []
        for col in input_features:
            for class_name in self.encoders[col].classes_:
                output_feature_names.append(f"{col}_{class_name}")
        
        return output_feature_names
    

class MultiLabelTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder for the high cardinality list features, using multi-label target encoding.
    """
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        """
        Fit the target encoder to the exploded data.
        """
        self.feature_names_in_ = X.columns.tolist()

        for col in X.columns:
            self.encoders[col] = TargetEncoder()
            exploded_series = X[col].explode()
            target_values = y.loc[exploded_series.index]
            self.encoders[col].fit(exploded_series.to_frame(), target_values)

        return self

    def transform(self, X):
        """
        Convert the list(s) to a one-hot-encoded matrix.
        """
        result = pd.DataFrame(index=X.index)

        for col in X.columns:
            exploded_series = X[col].explode()
            exploded_df = pd.DataFrame({col: exploded_series})
            encoded = self.encoders[col].transform(exploded_df)
            aggregated = encoded.groupby(level=0).mean()
            result[col] = aggregated.reindex(X.index, fill_value=0)[col]

        return result

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