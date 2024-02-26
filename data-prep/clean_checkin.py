# Load packages
import pandas as pd

# Read in data
file_path = "../data/check_in.html"
check_in = pd.read_html(file_path)
check_in_df = check_in[0]

# Clean data
check_in_df["Date"] = pd.to_datetime(check_in_df["Date"])
check_in_df = check_in_df.drop(columns=["Comment", "Status"])

# Save new data
check_in_df.to_pickle("../data/check_in.pkl")