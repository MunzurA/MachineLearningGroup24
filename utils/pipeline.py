import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from category_encoders import TargetEncoder

from ._feature_types import *
from ._custom_converters import PercentageConverter, ListConverter, BooleanConverter, DateConverter
from ._custom_encoders import MultiLabelOneHotEncoder, MultiLabelTargetEncoder
from ._config import *
    
#------------------------------------------------------------------------------

def create_pipeline(
        df: pd.DataFrame,
        model: BaseEstimator = DecisionTreeRegressor(random_state=RANDOM_STATE),
        filter: BaseEstimator = SelectKBest(f_regression, k=20),
        convert: bool = True,
        impute: bool = True,
        encode: bool = True,
        scale: bool = True,
        feature_selection: bool = True,
        model_selection: bool = True,
        ) -> Pipeline:
    """
    Creates a pipeline for preprocessing the given dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to preprocess.
        model (BaseEstimator): The model to be used in the pipeline. Default is DecisionTreeRegressor(random_state=24).
        filter (BaseEstimator): The filter to be used in the pipeline. Default is SelectKBest(f_regression, k=20).
        convert (bool): Whether to convert the features or not. Default is True.
        impute (bool): Whether to impute the missing values or not. Default is True.
        encode (bool): Whether to encode the categorical features or not. Default is True.
        scale (bool): Whether to scale the features or not. Default is True.
        feature_selection (bool): Whether to select the features or not. Default is True.
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
        steps.append(('scalers', _create_scalers(df)))
    if feature_selection:
        steps.append(('feature_selection', filter))
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
        n_jobs=-1,
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
        ).set_output(transform='pandas')


def _create_imputers(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of imputers for the given dataframe on the following feature types:
    - Numerical (iterative imputation)
    - Categorical (constant imputation: 'missing')
    - Text (constant imputation: '')

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
            IterativeImputer(
                initial_strategy='median',
                min_value=0,
                random_state=RANDOM_STATE,
            ),
            numerical_features
        ))

    # Categorical
    categorical_features = _check_feats(df, ORDER_FEATS + LOW_CARD_FEATS + HIGH_CARD_FEATS)
    if categorical_features:
        imputers.append((
            'impute_categorical',
            SimpleImputer(
                strategy='constant',
                fill_value="missing",
            ),
            categorical_features
        ))

    # Text
    text_features = _check_feats(df, TEXT_FEATS)
    if text_features:
        imputers.append((
            'impute_text',
            SimpleImputer(
                strategy='constant',
                fill_value='',
            ),
            text_features
        ))

    return ColumnTransformer(
        imputers,
        remainder='passthrough',
        n_jobs=-1,
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
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
                categories=[['within an hour', 'within a few hours', 'within a day', 'a few days or more', 'missing']],
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
                handle_unknown='value',
                handle_missing='value',
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
        n_jobs=-1,
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
        ).set_output(transform='pandas')


def _create_scalers(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer of scalers for the given dataframe on all features.

    Parameters:
        df (pd.DataFrame): The dataframe containing all the features.

    Returns:
        ColumnTransformer: The ColumnTransformer of scalers for the given dataframe.
    """
    scalers = []

    # Numerical (robust scaling)
    num_feats = _check_feats(df, NUM_FEATS + HIGH_CARD_FEATS + TEXT_FEATS + HIGH_CARD_LIST_FEATS + DATE_FEATS)
    if num_feats:
        scalers.append((
            'scale_numerical',
            RobustScaler(),
            num_feats
        ))

    return ColumnTransformer(
            scalers,
            remainder='passthrough',
            n_jobs=-1,
            verbose_feature_names_out=False,
            force_int_remainder_cols=False,
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