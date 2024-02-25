# Load packages
import pandas as pd

# Read in data
file_path = "../data/search.html"
search = pd.read_html(file_path)
search_df = search[0]

# Clean data
search_df["Date"] = pd.to_datetime(search_df["Date"])

# Save new data
search_df.to_pickle("../data/search.pkl")