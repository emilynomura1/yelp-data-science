# Load packages
import pandas as pd
from geopy.geocoders import Nominatim

# Define geocoder and load data
location = Nominatim(timeout=20, user_agent="yelp-ds-project")
reviews = pd.read_pickle("../data/user_review.pkl")

# Geocode from business name
reviews["Geocode"] = reviews["Business Name"].apply(location.geocode)

# Remove missing geocode entries
reviews.dropna(inplace=True)

# Extract coordinates
reviews['Latitude'] = [g.latitude for g in reviews["Geocode"]]
reviews['Longitude'] = [g.longitude for g in reviews["Geocode"]]

# Save new data
reviews.to_pickle("../data/reviews_geocoded.pkl")