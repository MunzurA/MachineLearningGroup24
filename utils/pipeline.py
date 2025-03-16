import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

from ._feature_types import *
from ._custom_converters import PercentageConverter, ListConverter, BooleanConverter, DateConverter
from ._custom_encoders import MultiLabelOneHotEncoder, MultiLabelTargetEncoder
    
#------------------------------------------------------------------------------

def create_pipeline(
        df: pd.DataFrame,
        model: BaseEstimator = DecisionTreeRegressor(random_state=24),
        convert: bool = True,
        impute: bool = True,
        encode: bool = True,
        scale: bool = True,
        model_selection: bool = True,
        ) -> Pipeline:
    """
    Creates a pipeline for preprocessing the given dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to preprocess.
        model (BaseEstimator): The model to be used in the pipeline. Default is LinearRegression().
        feature_selector (_univariate_selection._BaseFilter): The feature selector to be used in the pipeline. Default is SelectKBest(f_regression, k=20).
        convert (bool): Whether to convert the features or not. Default is True.
        impute (bool): Whether to impute the missing values or not. Default is True.
        encode (bool): Whether to encode the categorical features or not. Default is True.
        scale (bool): Whether to scale the features or not. Default is True.
        feature_selection (bool): Whether to add a feature selector or not. Default is True.
        model_selection (bool): Whether to add a model or not. Default is True.
    
    Returns:
        Pipeline: The pipeline for preprocessing the dataframe.
    """
    steps = []

    if convert:
        steps.append(('converters', _create_converters(df)))
    if impute:
        steps.append(('imputers', _create_imputers(df)))
    if encode:
        steps.append(('encoders', _create_encoders(df)))
    if scale:
        steps.append(('scalers', _create_scalers()))
    if model_selection:
        steps.append(('model_selection', model))
    
    return Pipeline(steps=steps)


def _create_converters(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of converters for the given dataframe on the following feature types:
    - Percentage (string to float)
    - List (as string to list)
    - Boolean (as string to boolean)
    - Date (as string to integer until a baseline date)

    Parameters:
        df (pd.DataFrame): The dataframe containing all the features.

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

    return ColumnTransformer(
        converters,
        remainder='passthrough',
        verbose_feature_names_out=False
        ).set_output(transform='pandas')


def _create_imputers(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of imputers for the given dataframe on the following feature types:
    - Numerical (median)
    - Categorical (constant: 'missing')
    - Text (constant: '')

    Parameters:
        df (pd.DataFrame): The dataframe containing all the features.

    Returns:
        ColumnTransformer: The ColumnTransformer of imputers for the given dataframe.
    """
    imputers = []

    # Numerical
    numerical_features = _check_feats(df, NUM_FEATS + PERC_FEATS + BOOL_FEATS + DATE_FEATS)
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

    return ColumnTransformer(
        imputers,
        remainder='passthrough',
        verbose_feature_names_out=False
        ).set_output(transform='pandas')


def _create_encoders(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of encoders for the given dataframe on the following feature types:
    - Categorical (ordinal encoding, one-hot encoding. and target encoding)
    - Text (TF-IDF vectorization)
    - List (target encoding)

    Parameters:
        df (pd.DataFrame): The dataframe containing all the features.

    Returns:
        ColumnTransformer: The ColumnTransformer of encoders for the given dataframe.
    """
    encoders = []

    # Categorical
    ordinal_features = _check_feats(df, ORDER_FEATS)
    if ordinal_features:
        encoders.append((
            'encode_ordinal',
            OrdinalEncoder(
                categories=[['within an hour', 'within a few hours', 'within a day', 'a few days or more']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            ordinal_features
        ))

    low_cardinality_features = _check_feats(df, LOW_CARD_FEATS)
    if low_cardinality_features:
        encoders.append((
            'encode_low_cardinality',
            OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                ),
            low_cardinality_features
        ))

    high_cardinality_features = _check_feats(df, HIGH_CARD_FEATS)
    if high_cardinality_features:
        encoders.append((
            'encode_high_cardinality',
            TargetEncoder(
                return_df=True,
                handle_unknown=-1,
            ),
            high_cardinality_features
        ))

    # List
    low_cardinality_list_features = _check_feats(df, LOW_CARD_LIST_FEATS)
    if low_cardinality_list_features:
        encoders.append((
            'encode_low_cardinality_list',
            MultiLabelOneHotEncoder(),
            low_cardinality_list_features
        ))

    high_cardinality_list_features = _check_feats(df, HIGH_CARD_LIST_FEATS)
    if high_cardinality_list_features:
        encoders.append((
            'encode_high_cardinality_list',
            MultiLabelTargetEncoder(),
            high_cardinality_list_features
        ))

    return ColumnTransformer(
        encoders,
        remainder='passthrough',
        verbose_feature_names_out=False
        ).set_output(transform='pandas')


def _create_scalers() -> ColumnTransformer:
    """
    Creates a ColumnTransformer of scalers for the given dataframe on all features.

    Returns:
        ColumnTransformer: The ColumnTransformer of scalers for the given dataframe.
    """
    return ColumnTransformer(
            [('scaler', StandardScaler(), slice(None))],
            remainder='passthrough',
            verbose_feature_names_out=False
            ).set_output(transform='pandas')

    
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