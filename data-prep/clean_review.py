# Load packages
import pandas as pd

# Read in data
file_path = "../data/user_review.html"
reviews = pd.read_html(file_path)
reviews_df = reviews[0]

# Clean data
reviews_df["Date"] = pd.to_datetime(reviews_df["Date"])
reviews_df = reviews_df.drop(columns=["IP Address","Status"])

# Save new data
reviews_df.to_pickle("../data/user_review.pkl")