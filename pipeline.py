import pandas as pd

from datetime import datetime
from ast import literal_eval

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MultiLabelBinarizer
from category_encoders import TargetEncoder


#------------------------------------------------------------------------------

# Inherent ordered features (ordinal encoding)
ORDER_FEAT = [
    'host_response_time',
]

# Low-Cardinality features (one-hot encoding)
LOW_CARD_FEATS = [
    'source',
    'host_response_time',
    'neighbourhood',
    'neighbourhood_group_cleansed',
    'property_type',
]

# High-Cardinality features (target encoding)
HIGH_CARD_FEATS = [
    'host_name',
    'host_location',
    'host_neighbourhood',
    'neighbourhood_cleansed',
    'property_type',
]

# Numerical features (scaling)
NUM_FEATS = [
    'host_listings_count',
    'host_total_listings_count',
    'latitude',
    'longitude',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'minimum_nights',
    'maximum_nights',
    'minimum_minimum_nights',
    'maximum_minimum_nights',
    'minimum_maximum_nights',
    'maximum_maximum_nights',
    'minimum_nights_avg_ntm',
    'maximum_nights_avg_ntm',
    'availability_30',
    'availability_60',
    'availability_90',
    'availability_365',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'number_of_reviews_l30d',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'calculated_host__listings_count',
    'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms',
    'reviews_per_month',
]

# Text features (tokenization)
TEXT_FEATS = [
    'name',
    'description',
    'neighborhood_overview',
    'host_about',
    'bathrooms_text',
]

# Special features (requiring custom transformers)
PRICE_FEATS = [
    'price',
]
PERC_FEATS = [
    'host_response_rate',
    'host_acceptance_rate',
]
LOW_CARD_LIST_FEATS = [
    'host_verifications',
]
HIGH_CARD_LIST_FEATS = [
    'amenities',
]
BOOL_FEATS = [
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'has_availability',
    'instant_bookable',
]
DATE_FEATS = [
    'last_scraped',
    'host_since',
    'calendar_updated',
    'calendar_last_scraped',
    'first_review',
    'last_review',
]

# Unwanted features (cannot be transformed)
UNWANTED_FEATS = [
    'id',
    'listing_url',
    'scrape_id',
    'picture_url',
    'host_id',
    'host_url',
    'host_thumbnail_url',
    'host_picture_url',
    'license',
]

#------------------------------------------------------------------------------

class PriceConverter(BaseEstimator, TransformerMixin):
    """
    Custom converter for the price feature.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace('$', '').str.replace(',', '').astype(float)
    
#------------------------------------------------------------------------------

def create_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Creates a pipeline for preprocessing the given dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to preprocess.
    
    Returns:
        Pipeline: The pipeline for preprocessing the dataframe.
    """
    transformers = []

    # Inherent ordered features (ordinal encoding)
    ordinal_features = __check_feats(df, ORDER_FEAT)
    for feat in ordinal_features:
        entries = df[feat].dropna().unique()

        transformers.append((f'{feat}_ordinal', Pipeline([
            ('encoder', OrdinalEncoder(categories=entries))
            ]), [feat]))

    # Low-Cardinality features (one-hot encoding)
    low_card_features = __check_feats(df, LOW_CARD_FEATS)
    for feat in low_card_features:
        entries = df[feat].dropna().unique()

        transformers.append((f'{feat}_low_card', Pipeline([
            ('encoder', OneHotEncoder(categories=entries))
        ]), [feat]))

    # High-Cardinality features (target encoding)
    high_card_features = __check_feats(df, HIGH_CARD_FEATS)
    if high_card_features:
        transformers.append(('high_card', Pipeline([
            ('encoder', TargetEncoder())
        ]), high_card_features))

    # Numerical features (scaling)
    num_features = __check_feats(df, NUM_FEATS)
    if num_features:
        transformers.append(('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_features))

    # Text features (tokenization)
    text_features = __check_feats(df, TEXT_FEATS)
    if text_features:
        transformers.append(('text', Pipeline([
            ('vectorizer', TfidfVectorizer(lowercase=True, max_features=100))
        ]), text_features))

    # Price feature (converting to numerical)
    price_features = __check_feats(df, PRICE_FEATS)
    if price_features:
        transformers.append(('price', Pipeline([
            ('converter', PriceConverter()),
            ('imputer', SimpleImputer(strategy='mean')),
        ]), price_features))

    # Percentage features (converting to numerical)

    # Low cardinality list features (multi label binary encoding)

    # High cardinality list features (multi label binary encoding)

    # Boolean mapping ('t'/'f' -> 1/0)

    # Date features (converting to numerical difference between entry and baseline)

    # Combine all transformers into a single pipeline
    pipeline = Pipeline([
        ('preprocessing', ColumnTransformer(transformers))
    ])

    


def __check_feats(df: pd.DataFrame, feats: list) -> list:
    """
    Helper function to check if the given features are present in the dataframe and return them.

    Parameters:
        df (pd.DataFrame): The dataframe to check.
        feats (list): The features to check.

    Returns:
        list : The features that are present in the dataframe.
    """
    return [feature for feature in feats if feature in df.columns]