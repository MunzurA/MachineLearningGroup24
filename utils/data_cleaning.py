import pandas as pd
import numpy as np

#------------------------------------------------------------

def load_and_clean(**kwargs) -> pd.DataFrame:
    """
    Cleans the raw data by dropping unnecessary columns and converting the target price column to a float.

    Parameters:
        **kwargs: Additional keyword arguments to pass to the drop_features and handle_price functions.

    Returns:
        pd.DataFrame: The cleaned data as a pandas DataFrame.
    """
    data = read_data()
    data = drop_features(data, **kwargs)
    data = drop_rows(data, **kwargs)
    
    return data


def drop_rows(data: pd.DataFrame, quantile: float = 0.99, verbose: bool = False, **kwargs) -> pd.DataFrame:
    """
    Handles the price feature by converting it to a float and handling missing values and outliers.

    Parameters:
        data (pd.DataFrame): The data to handle the price feature for.
        quantile (float): The quantile to use as the threshold for outlier removal. Deffaults to 0.99.
        verbose (bool): Whether to print the number of rows removed due to missing price and outliers. Default is False.
        **kwargs: Additional keyword arguments from other functions.

    Returns:
        pd.DataFrame: The data with the price feature handled.
    """
    num_rows_before = len(data)

    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)

    if verbose:
        print(f"Number of outliers removed above quantile {quantile} (${data['price'].quantile(quantile)}): {len(data[data['price'] > data['price'].quantile(quantile)])}")
        print(f"number of rows removed due to missing price: {data['price'].isna().sum()}")
    data = data[data['price'] < data['price'].quantile(quantile)]

    if verbose:
        print(f"Number of rows left after cleaning ({num_rows_before} - {num_rows_before - len(data)}): {len(data)}")

    return data


def read_data() -> pd.DataFrame:
    """
    Reads the raw data from the csv gzip file and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The raw data as a pandas DataFrame.
    """
    # Load the raw data from the csv gzip file
    return pd.read_csv('data/listings.csv.gz', index_col=0, compression='gzip')


def drop_features(
        data: pd.DataFrame,
        remove_redundant_feats: bool = True,
        manual_redundant_feats: list = [
            'listing_url',
            'scrape_id',
            'last_scraped',
            'source',
            'name',
            'description',
            'neighborhood_overview',
            'picture_url',
            'host_id',
            'host_url',
            'host_name',
            'host_location',
            'host_about',
            'host_thumbnail_url',
            'host_picture_url',
            'host_neighbourhood',
            'calendar_last_scraped',
            ],
        add_redundant_feats: list = [],
        remove_high_corr_feats: bool = True,
        corr_threshold: float = 0.8,
        remove_missing_feats: bool = True,
        missing_threshold: float = 0.5,
        remove_single_value_feats: bool = True,
        verbose: bool = False,
        **kwargs
        ) -> pd.DataFrame:
    """
    Drops the columns that are not relevant to the analysis.
    Specifically:
    1. Those which are not relevant to the analysis.
    2. Those which have a high correlation to another column.
    3. Those which hold more than a specified percentage of missing values.
    4. Those with only one uniue value.

    Parameters:
        df (pd.DataFrame) : The dataframe to be cleaned.

    Returns:
        pd.DataFrame : The cleaned dataframe.
    """
    df = data.copy()

    if remove_redundant_feats:
        # Drop any features which are not elevant to the analysis
        to_remove = np.unique(manual_redundant_feats + add_redundant_feats)
        df = df.drop(to_remove, axis=1)
        if verbose:
            print(f"Features dropped due to redundancy:\n{to_remove}\n")

    if remove_high_corr_feats:
        # Drop any features which have a high correlation to another column
        corr = df.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        high_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        df = df.drop(high_corr, axis=1)
        if verbose:
            print(f"Features dropped due to high correlation (>{corr_threshold}):\n{high_corr}\n")

    if remove_missing_feats:
        # Drop any features which have more than a specified percentage of missing values
        missing = df.isnull().mean()
        to_remove = missing[missing > missing_threshold].index.tolist()
        df = df.drop(to_remove, axis=1)
        if verbose:
            print(f"Features dropped due to amount of missing values (>{missing_threshold * 100}%):\n{to_remove}\n")

    if remove_single_value_feats:
        # Drop any features which have only one unique value
        single_value = df.nunique() == 1
        to_remove = single_value[single_value].index.tolist()
        df = df.drop(to_remove, axis=1)
        if verbose:
            print(f"Features dropped due to only one unique value:\n{to_remove}\n")

    return df
