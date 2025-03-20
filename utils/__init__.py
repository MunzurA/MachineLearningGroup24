import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#------------------------------------------------------------------------------

def _get_feature_names_from_column_transformer(column_transformer: ColumnTransformer) -> list:
    """
    Retrieve the feature names from a ColumnTransformer in order to reconstruct the dataframe after a conversion.

    Parameters:
        column_transformer (ColumnTransformer): The ColumnTransformer to retrieve the feature names from.

    Returns:
        list: The feature names.
    """
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        # Skip dropped columns
        if name == 'drop' or transformer == 'drop':
            continue

        # Handle passthrough or remainder columns
        if transformer == 'passthrough' or transformer == 'remainder':
            if hasattr(column_transformer, 'feature_names_in_'):
                feature_names.extend(column_transformer.feature_names_in_[columns])
            else:
                print(f"Warning: column_transformer does not have feature_names_in_ attribute.")
            continue

        # Get feature names from transformer (regular or encoded)
        if hasattr(transformer, 'get_feature_names_out'):
            trans_feature_names = transformer.get_feature_names_out()
        elif hasattr(transformer, 'categories_'):
            trans_feature_names = []
            for i, category in enumerate(transformer.categories_):
                for cat in category:
                    trans_feature_names.append(f"{columns[i]}_{cat}")
        else:
            if isinstance(columns, list):
                trans_feature_names = columns
            else:
                trans_feature_names = [columns]

        feature_names.extend(trans_feature_names)

    return feature_names


def convert_df(df: pd.DataFrame, pipeline: Pipeline, target: str = 'price') -> pd.DataFrame:
    """
    Convert a DataFrame to a DataFrame with transformed features.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be converted.
        pipeline (Pipeline): The pipeline containing the feature transformation steps.
        target (str): The target column name.

    Returns:
        pd.DataFrame: A DataFrame with transformed features.
    """
    # Extract the target variable and features
    X = df.drop(columns=[target])
    y = df[target]

    # Fit the pipeline and transform the data
    X_transformed = pipeline.fit_transform(X, y)

    # Get the feature names from the converter ColumnTransformer
    column_transformer = pipeline.named_steps['converters']
    feature_names = _get_feature_names_from_column_transformer(column_transformer)

    return pd.DataFrame(X_transformed, columns=feature_names)
