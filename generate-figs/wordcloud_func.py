# Load packages
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Generate wordcloud from a review
stop_words = set(stopwords.words('english'))
def get_wordcloud(review, fig_name):
    word_tokens = word_tokenize(review)
    filtered_sentence = [word.lower() for word in word_tokens if word not in stop_words]
    sentence = ' '.join(filtered_sentence)
    wordcloud = WordCloud(width=500, 
                          height=300, 
                          max_words=50, 
                          stopwords=stop_words, 
                          colormap='Blues').generate_from_text(sentence)
    fig_path = '../figures/' + fig_name + '_wordcloud.png'
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(fig_path, bbox_inches='tight')