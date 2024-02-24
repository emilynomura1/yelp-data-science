import pandas as pd
from geopy.geocoders import Nominatim

# Uses current IP address to aid in geocoding addresses
location = Nominatim(timeout=20, user_agent="yelp-ds-project")
reviews = pd.read_pickle("../data/user_review.pkl")

reviews["Geocode"] = reviews["Business Name"].apply(location.geocode)

# Remove missing geocode entries to avoid error when extracting coordinates
reviews.dropna(inplace=True)

reviews['Latitude'] = [g.latitude for g in reviews["Geocode"]]
reviews['Longitude'] = [g.longitude for g in reviews["Geocode"]]

reviews.to_pickle("../data/reviews_geocoded.pkl")