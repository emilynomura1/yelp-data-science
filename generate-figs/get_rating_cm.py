# Loack packages
import pandas as pd
import numpy as np
from nltk import word_tokenize
import matplotlib.pyplot as plt
from sklearn import metrics

# Load & clean data
sentiment_results = pd.read_pickle("../results/bert_reviews_sentiment.pkl")
reviews = pd.read_pickle("../data/user_review.pkl")
results_list = []
for i in sentiment_results:
    results_list.append(i[0][0]) # Grab highest probability of predicted rating
results_df = pd.DataFrame.from_dict(results_list)
# Drop reviews that were removed during the sentiment analysis
review_lengths = []
for review in reviews["Comment"]:
    review_lengths.append(len(word_tokenize(review)))
ind_remove = np.where(np.asarray(review_lengths)>=400)[0] # Indices we want to remove
reviews.drop(index=ind_remove, inplace=True)
reviews.reset_index(drop=True, inplace=True)
# Combine reviews data frame with predicted ratings
sentiment_df = pd.concat([reviews, results_df], axis=1)
sentiment_df["label_clean"] = sentiment_df["label"].str.get(0).astype(int)

# Confusion matrix between actuals & predictions
conf_matrix = metrics.confusion_matrix(sentiment_df["Rating"], sentiment_df["label_clean"])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[1,2,3,4,5])
cm_display.plot()
plt.savefig("../figures/ratings_cm.png", bbox_inches='tight')