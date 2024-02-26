# Load packages
import pandas as pd
from transformers import pipeline
from nltk import word_tokenize
import pickle

# Initialize pre-trained sentiment classifier
# Source: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
bert_sentiment_classifier = pipeline(
   model="nlptown/bert-base-multilingual-uncased-sentiment", 
   top_k=None
)

# Load and prep data
reviews = pd.read_pickle("../data/user_review.pkl")
reviews_list = reviews["Comment"].to_list()
# Remove sentences that are too long for this model
tokenized_reviews = []
for review in reviews_list:
    if len(word_tokenize(review)) < 400:
        tokenized_reviews.append(review)

# Save classifications
results_list = []
for review in tokenized_reviews:
    results_list.append(bert_sentiment_classifier(review))
with open("../results/bert_reviews_sentiment.pkl", 'wb') as f:
    pickle.dump(results_list, f)