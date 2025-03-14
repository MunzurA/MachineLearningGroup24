import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, _univariate_selection
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

from ._feature_types import *
from ._custom_converters import PercentageConverter, ListConverter, BooleanConverter, DateConverter
    
#------------------------------------------------------------------------------

def create_pipeline(df: pd.DataFrame, model: BaseEstimator = LinearRegression(), feature_selection: _univariate_selection._BaseFilter = SelectKBest(chi2, k=20)) -> Pipeline:
    """
    Creates a pipeline for preprocessing the given dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to preprocess.
        model (BaseEstimator): The model to be used in the pipeline.
    
    Returns:
        Pipeline: The pipeline for preprocessing the dataframe.
    """
    converters = _create_converters(df)
    imputers = _create_imputers(df)
    encoders = _create_encoders(df)
    scalers = _create_scalers(df)
    
    return Pipeline(steps=[
        ('converters', ColumnTransformer(converters)),
        ('imputers', ColumnTransformer(imputers)),
        ('encoders', ColumnTransformer(encoders)),
        ('scalers', ColumnTransformer(scalers)),
        ('feature_selector', feature_selection),
        ('model', model)
    ])


def _create_converters(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of converters for the given dataframe on the following feature types:
    - Percentage (string to float)
    - List (as string to list)
    - Boolean (as string to boolean)
    - Date (as string to integer until a baseline date)

    Returns:
        ColumnTransformer: The ColumnTransformer of converters for the given dataframe.
    """
    converters = []

    # Percentages
    percentage_features = _check_feats(df, PERC_FEATS)
    if percentage_features:
        converters.append((
            'convert_percentages',
            PercentageConverter(),
            percentage_features
        ))

    # Lists
    list_features = _check_feats(df, LOW_CARD_LIST_FEATS + HIGH_CARD_LIST_FEATS)
    if list_features:
        converters.append((
            'convert_lists',
            ListConverter(),
            list_features
        ))

    # Booleans
    boolean_features = _check_feats(df, BOOL_FEATS)
    if boolean_features:
        converters.append((
            'convert_booleans',
            BooleanConverter(),
            boolean_features
        ))

    # Dates
    date_features = _check_feats(df, DATE_FEATS)
    if date_features:
        converters.append((
            'converter',
            DateConverter(),
            date_features
        ))

    return ColumnTransformer(converters)


def _create_imputers(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of imputers for the given dataframe on the following feature types:
    - Numerical (median)
    - Categorical (constant: 'missing')
    - Text (constant: '')
    """
    imputers = []

    # Numerical
    numerical_features = _check_feats(df, NUM_FEATS)
    if numerical_features:
        imputers.append((
            'impute_numerical',
            SimpleImputer(strategy='median'),
            numerical_features
        ))

    # Categorical
    categorical_features = _check_feats(df, ORDER_FEATS + LOW_CARD_FEATS + HIGH_CARD_FEATS)
    if categorical_features:
        imputers.append((
            'impute_categorical',
            SimpleImputer(strategy='constant', fill_value='missing'),
            categorical_features
        ))

    # Text
    text_features = _check_feats(df, TEXT_FEATS)
    if text_features:
        imputers.append((
            'impute_text',
            SimpleImputer(strategy='constant', fill_value=''),
            text_features
        ))

    return ColumnTransformer(imputers)


def _create_encoders(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of encoders for the given dataframe on the following feature types:
    - Categorical (ordinal encoding, one-hot encoding. and target encoding)
    - Text (TF-IDF vectorization)
    - List (target encoding)
    """
    encoders = []

    # Categorical
    ordinal_features = _check_feats(df, ORDER_FEATS)
    if ordinal_features:
        encoders.append((
            'encode_ordinal',
            OrdinalEncoder(),
            ordinal_features
        ))

    low_cardinality_features = _check_feats(df, LOW_CARD_FEATS)
    if low_cardinality_features:
        encoders.append((
            'encode_low_cardinality',
            OneHotEncoder(handle_unknown='ignore'),
            low_cardinality_features
        ))

    high_cardinality_features = _check_feats(df, HIGH_CARD_FEATS)
    if high_cardinality_features:
        encoders.append((
            'encode_high_cardinality',
            TargetEncoder(),
            high_cardinality_features
        ))

    # Text
    text_features = _check_feats(df, TEXT_FEATS)
    if text_features:
        encoders.append((
            'encode_text',
            TfidfVectorizer(),
            text_features
        ))

    # List
    low_cardinality_list_features = _check_feats(df, LOW_CARD_LIST_FEATS)
    if low_cardinality_list_features:
        encoders.append((
            'encode_low_cardinality_list',
            #TODO,
            low_cardinality_list_features
        ))

    high_cardinality_list_features = _check_feats(df, HIGH_CARD_LIST_FEATS)
    if high_cardinality_list_features:
        encoders.append((
            'encode_high_cardinality_list',
            #TODO,
            high_cardinality_list_features
        ))

    return ColumnTransformer(encoders)


def _create_scalers(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of scalers for the given dataframe on the following feature types:
    - Numerical (standard scaling)
    """
    scalers = []

    numerical_features = _check_feats(df, NUM_FEATS + PERC_FEATS + DATE_FEATS)
    if numerical_features:
        scalers.append((
            'scale_numerical',
            StandardScaler(),
            numerical_features
        ))

    return ColumnTransformer(scalers)

    
def _check_feats(df: pd.DataFrame, feats: list) -> list:
    """
    Helper function to check if the given features are present in the dataframe and return them.

    Parameters:
        df (pd.DataFrame): The dataframe to check.
        feats (list): The features to check.

    Returns:
        list : The features that are present in the dataframe.
    """
    return [feature for feature in feats if feature in df.columns]