# Load packages
import pandas as pd
from wordcloud_func import get_wordcloud

# Load data
reviews = pd.read_pickle("../data/user_review.pkl")
search = pd.read_pickle("../data/search.pkl")

# Clean search data
search_list = []
for i in search["Search Text"]:
    if pd.isna(i) == False:
        search_list.append(i)
search_string = ' '.join(search_list)

# Get wordcloud for a review
get_wordcloud(reviews["Comment"][100], "review")

# Get wordcloud for searches
get_wordcloud(search_string, "searches")