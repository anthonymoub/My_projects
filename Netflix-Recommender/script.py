 
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import requests

 
# Quick look at dataset

df = pd.read_csv("data/netflix_titles.csv")

 
# Since the recommendation system to be built is a text based, we should do some text pre processing amd cleaning

# This function uses the re.sub function from the re (regular expressions) module to replace all characters in the string that are not letters (i.e. not a-z or A-Z) with spaces

@st.cache_resource
def clean_desc(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'[^a-zA-Z]', ' ', s) 
    return s

# Apply the above function to the Description column 
df['description'] = df['description'].apply(clean_desc)

# Tokenizing the words for lemmatization and removing stopwords
df['description'] = df['description'].apply(word_tokenize)
df['description'] = df['description'].apply(
  lambda x:[word for word in x if word not in set(stopwords.words('english'))]
)

# Joining the words after lemmatization and stopword removal
df['description'] = df['description'].apply(lambda x: ' '.join(x))


 
# Use TF-IDF to convert words into vectors that could be used in modeling

# Making an object of TfidfVectorizer in which words contains only in 1 document and word repeated in 70% of documents are ignored. These words do not have that much information gain
tfidf = TfidfVectorizer(min_df = 2, max_df = 0.7)

# Fitting the cleaned text in TfidfVectorizer

X = tfidf.fit_transform(df['description'])


# making a suitable dataframe for calculating the cosine similarity and save it
tfidf_df = pd.DataFrame(X.toarray(), columns = tfidf.get_feature_names())
tfidf_df.index = df['title']
 
 
# Cosine similarity function
@st.cache_resource
def recommend_table(list_of_movie_enjoyed, tfidf_data, movie_count=20):
    """
    function for recommending movies
    :param list_of_movie_enjoyed: list of movies
    :param tfidf_data: self-explanatory
    :param movie_count: no of movies to suggest
    :return: list of suggested movie titles
    """
    movie_enjoyed_df = tfidf_data.reindex(list_of_movie_enjoyed)
    user_prof = movie_enjoyed_df.select_dtypes('number').mean()  # select only numeric columns
    tfidf_subset_df = tfidf_data.drop(list_of_movie_enjoyed)
    # select only numeric columns for the subset as well
    tfidf_subset_df = tfidf_subset_df.select_dtypes('number') 
    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).head(movie_count)

    recommended_indices = sorted_similarity_df.index.tolist()
    recommended_titles = tfidf_data.loc[recommended_indices, 'title'].tolist()

    return recommended_titles


 
@st.cache_resource
def get_movie_posters(movie_titles, api_key):
    """
    function for getting movie posters using OMDB API
    :param movie_titles: list of movie titles
    :param api_key: API key for OMDB API
    :return: list of movie posters
    """
    base_url = "http://www.omdbapi.com/"
    posters = []
    for title in movie_titles:
        params = {"apikey": api_key, "t": title, "type": "movie"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json()
            if results.get("Poster") != "N/A":
                posters.append(results.get("Poster"))
            else:
                posters.append(None)
        else:
            posters.append(None)
    return posters

 
# Convert movie titles to pickle file to be used for building the application's dropdown

titles = df['title']
import pickle

with open('data/titles.pkl', 'wb') as f:
    pickle.dump(titles, f)


