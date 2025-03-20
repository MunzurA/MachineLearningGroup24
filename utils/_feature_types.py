# Inherent ordered features (ordinal encoding)
ORDER_FEATS = [
    'host_response_time', # ()
]

# Low-Cardinality features (one-hot encoding)
LOW_CARD_FEATS = [
    'source', # (text) One of "neighbourhood search" or "previous scrape". "neighbourhood search" means that the listing was found by searching the city, while "previous scrape" means that the listing was seen in another scrape performed in the last 65 days, and the listing was confirmed to be still available on the Airbnb site.
    'neighbourhood', # (text)
    'neighbourhood_group_cleansed', # (text) The neighbourhood group as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
    'room_type', # (text) "[Entire home/apt|Private room|Shared room|Hotel] All homes are grouped into the following three room types: Entire place Private room Shared room Entire place Entire places are best if you're seeking a home away from home. With an entire place, you'll have the whole space to yourself. This usually includes a bedroom, a bathroom, a kitchen, and a separate, dedicated entrance. Hosts should note in the description if they'll be on the property or not (ex: ""Host occupies first floor of the home""), and provide further details on the listing. Private rooms Private rooms are great for when you prefer a little privacy, and still value a local connection. When you book a private room, you'll have your own private room for sleeping and may share some spaces with others. You might need to walk through indoor spaces that another host or guest may occupy to get to your room. Shared rooms Shared rooms are for when you don't mind sharing a space with others. When you book a shared room, you'll be sleeping in a space that is shared with others and share the entire space with other people. Shared rooms are popular among flexible travelers looking for new friends and budget-friendly stays."
]

# High-Cardinality features (target encoding)
HIGH_CARD_FEATS = [
    'host_name', # (text) Name of the host. Usually just the first name(s).
    'host_location', # (text) The host's self reported location.
    'host_neighbourhood', # (text)
    'neighbourhood_cleansed', # (text) The neighbourhood as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.
    'property_type', # (text) Self selected property type. Hotels and Bed and Breakfasts are described as such by their hosts in this field
]

# Numerical features (scaling)
NUM_FEATS = [
    'host_listings_count', # (text) The number of listings the host has (per Airbnb unknown calculations)
    'host_total_listings_count', # (text) The number of listings the host has (per Airbnb unknown calculations)
    'latitude', # (numeric) Uses the World Geodetic System (WGS84) projection for latitude and longitude.
    'longitude', # (numeric) Uses the World Geodetic System (WGS84) projection for latitude and longitude.
    'accommodates', # (integer) The maximum capacity of the listing
    'bathrooms', # (numeric) The number of bathrooms in the listing
    'bedrooms', # (integer) The number of bedrooms
    'beds', # (integer) The number of bed(s)
    'minimum_nights', # (integer) minimum number of night stay for the listing (calendar rules may be different)
    'maximum_nights', # (integer) maximum number of night stay for the listing (calendar rules may be different)
    'minimum_minimum_nights', # (integer) the smallest minimum_night value from the calender (looking 365 nights in the future)
    'maximum_minimum_nights', # (integer) the largest minimum_night value from the calender (looking 365 nights in the future)
    'minimum_maximum_nights', # (integer) the smallest maximum_night value from the calender (looking 365 nights in the future)
    'maximum_maximum_nights', # (integer) the largest maximum_night value from the calender (looking 365 nights in the future)
    'minimum_nights_avg_ntm', # (numeric) the average minimum_night value from the calender (looking 365 nights in the future)
    'maximum_nights_avg_ntm', # (numeric) the average maximum_night value from the calender (looking 365 nights in the future)
    'availability_30', # (integer) avaliability_x. The availability of the listing x days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.
    'availability_60', # (integer) avaliability_x. The availability of the listing x days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.
    'availability_90', # (integer) avaliability_x. The availability of the listing x days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.
    'availability_365', # (integer) avaliability_x. The availability of the listing x days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.
    'number_of_reviews', # (integer) The number of reviews the listing has
    'number_of_reviews_ltm', # (integer) The number of reviews the listing has (in the last 12 months)
    'number_of_reviews_l30d', # (integer) The number of reviews the listing has (in the last 30 days)
    'review_scores_rating', # ()
    'review_scores_accuracy', # ()
    'review_scores_cleanliness', # ()
    'review_scores_checkin', # ()
    'review_scores_communication', # ()
    'review_scores_location', # ()
    'review_scores_value', # ()
    'calculated_host__listings_count', # (integer) The number of listings the host has in the current scrape, in the city/region geography.
    'calculated_host_listings_count_entire_homes', # (integer) The number of Entire home/apt listings the host has in the current scrape, in the city/region geography
    'calculated_host_listings_count_private_rooms', # (integer) The number of Private room listings the host has in the current scrape, in the city/region geography
    'calculated_host_listings_count_shared_rooms', # (integer) The number of Shared room listings the host has in the current scrape, in the city/region geography
    'reviews_per_month', # (numeric) The average number of reviews per month the listing has over the lifetime of the listing. Psuedocoe/~SQL: IF scrape_date - first_review <= 30 THEN number_of_reviews ELSE number_of_reviews / ((scrape_date - first_review + 1) / (365/12))
]

# Text features (tokenization)
TEXT_FEATS = [
    'name', # (text) Name of the listing
    'description', # (text) Detailed description of the listing
    'neighborhood_overview', # (text) Host's description of the neighbourhood
    'host_about', # (text) Description about the host
    'bathrooms_text', # (string) "The number of bathrooms in the listing. On the Airbnb web-site, the bathrooms field has evolved from a number to a textual description. For older scrapes, bathrooms is used."
]

# Special features (requiring custom transformers)
PRICE_FEATS = [
    'price', # (currency) "daily price in local currency. NOTE: the $ sign is a technical artifact of the export, please ignore it"
]
PERC_FEATS = [
    'host_response_rate', # ()
    'host_acceptance_rate', # () That rate at which a host accepts booking requests.
]
LOW_CARD_LIST_FEATS = [
    'host_verifications', # ()
]
HIGH_CARD_LIST_FEATS = [
    'amenities', # (json)
]
BOOL_FEATS = [
    'host_is_superhost', # (boolean[t=true; f=false])
    'host_has_profile_pic', # (boolean[t=true; f=false])
    'host_identity_verified', # (boolean[t=true; f=false])
    'has_availability', # (boolean) [t=true; f=false]
    'instant_bookable', # (boolean) [t=true; f=false]. Whether the guest can automatically book the listing without the host requiring to accept their booking request. An indicator of a commercial listing.
]
DATE_FEATS = [
    'last_scraped', # (datetime) UTC. The date and time this listing was "scraped"
    'host_since', # (date) The date the host/user was created. For hosts that are Airbnb guests this could be the date they registered as a guest.
    'calendar_updated', # (date)
    'calendar_last_scraped', # (date)
    'first_review', # (date) The date of the first/oldest review
    'last_review', # (date) The date of the last/newest review
]

# Unwanted features (cannot be transformed)
UNWANTED_FEATS = [
    'id', # (integer) Airbnb's unique identifier for the listing
    'listing_url', # (text)
    'scrape_id', # (bigint) Inside Airbnb "Scrape" this was part of
    'picture_url', # (text) URL to the Airbnb hosted regular sized image for the listing
    'host_id', # (integer) Airbnb's unique identifier for the host/user
    'host_url', # (text) The Airbnb page for the host
    'host_thumbnail_url', # (text)
    'host_picture_url', # (text)
    'license', # (text) The licence/permit/registration number
]