###
### Function to transform a review column to a list of tokens
###

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def token_list(df, col):

  all_reviews = df[col].str.cat(sep=' ') # concatenates all reviews
  all_reviews = all_reviews.lower() # lowercase
  all_reviews = re.sub('[^A-Za-z]+', ' ',all_reviews) # remove punctuation

  reviews_tokens = word_tokenize(all_reviews) # tokenize (NLTK)
  reviews_tokens = [t for t in reviews_tokens if t not in stopwords.words('english')] # remove stopwords (NLTK)
  reviews_tokens = [t for t in reviews_tokens if len(t) >2] # remove tokens of less than 3 characters

  lemmatizer = WordNetLemmatizer()
  reviews_tokens = [lemmatizer.lemmatize(t) for t in reviews_tokens] # lemmatisation

  return reviews_tokens


###
### Function to plot a wordcloud given a tokens list
###

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def cloud_plot(tokens):
  
  fig = plt.figure(figsize = (10,5))
  wc = WordCloud(width=600,height=300,background_color = 'black', colormap='coolwarm', max_words = 100,  max_font_size = 100)
  wc.generate(' '.join(tokens))
  plt.imshow(wc)
  plt.axis('off')
  plt.close(fig)
  return fig


###
### Function to save a token list as a dataframe that captures tokens' frequency
###

import nltk
import pandas as pd

def tokens_freq_df(tokens):

  tokens_freq = nltk.FreqDist(tokens) # tokens distribution
  df = (pd.DataFrame(tokens_freq.items(), columns=["token", "frequency"]).sort_values("frequency", ascending=False).reset_index(drop=True))
  return df


###
### Function to build a BERTopic model
###

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def build_topic_model(embedder, SEED):
    return BERTopic(
        embedding_model=embedder,   # previously defined when clustering conditions
        umap_model=UMAP(
            n_neighbors=15, 
            n_components=5,
            min_dist=0.02, 
            metric="cosine", 
            random_state=SEED
            ),

        hdbscan_model=HDBSCAN(
            min_cluster_size=5,
            min_samples=2
            ),      

        vectorizer_model=CountVectorizer(
            stop_words="english", 
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            min_df=2, 
            ngram_range=(1,2)
            )
    )


###
### Function to fix specific patterns in topics from LLM output
###

def fix_topic_list(entry):
    blob = ' '.join(entry)                       # combine whatever fragments are there
    extracted = re.findall(r'"([^"]+)"', blob)    # pull out anything between quote pairs
    if extracted:
        return extracted
    return entry   # nothing quoted found — already clean, leave untouched