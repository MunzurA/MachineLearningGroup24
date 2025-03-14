# Inherent ordered features (ordinal encoding)
ORDER_FEATS = [
    'host_response_time',
]

# Low-Cardinality features (one-hot encoding)
LOW_CARD_FEATS = [
    'source',
    'host_response_time',
    'neighbourhood',
    'neighbourhood_group_cleansed',
    'room_type',
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