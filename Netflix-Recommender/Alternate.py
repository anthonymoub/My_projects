# import libraries
import pandas as pd
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit import session_state as session
from script import recommend_table

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_data():
    """
    load and cache data
    :return: tfidf data
    """
    tfidf_data = pd.read_csv("data/tfidf_data.csv", index_col=0)
    return tfidf_data

# load the data
df = pd.read_csv("data/netflix_titles.csv")

# clean the data
df['description'] = df['description'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', str(x)).lower())
df['description'] = df['description'].apply(word_tokenize)
df['description'] = df['description'].apply(lambda x: [word for word in x if word not in set(stopwords.words('english'))])
df['description'] = df['description'].apply(lambda x: ' '.join(x))

# convert text to vector using TF-IDF
tfidf = TfidfVectorizer(min_df=2, max_df=0.7)
X = tfidf.fit_transform(df['description'])

# save the data to a file
tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names())
tfidf_df.index = df['title']
tfidf_df.to_csv("data/tfidf_data.csv")

# define the recommender function
def recommend_table(list_of_movie_enjoyed, tfidf_data, movie_count=20):
    """
    function for recommending movies
    :param list_of_movie_enjoyed: list of movies
    :param tfidf_data: self-explanatory
    :param movie_count: no of movies to suggest
    :return: dataframe containing suggested movie
    """
    movie_enjoyed_df = tfidf_data.reindex(list_of_movie_enjoyed)
    user_prof = movie_enjoyed_df.mean()
    tfidf_subset_df = tfidf_data.drop(list_of_movie_enjoyed)
    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    sorted_similarity_df = similarity_df.nlargest(movie_count, 'similarity_score')

    return sorted_similarity_df

# load the titles to a variable
with open('data/titles.pkl', 'rb') as f:
    titles = pickle.load(f)

# build the UI
st.title("Netflix Recommendation System")
st.text("This is an Content Based Recommender System made on implicit ratings :smile:.")
session.options = st.multiselect(label="Select Movies", options=titles)
session.slider_count = st.slider(label="movie_count", min_value=5, max_value=50)
is_clicked = st.button(label="Recommend")

# perform the recommendation and display the results
if is_clicked:
    tfidf_data = load_data()
    dataframe = recommend_table(session.options, movie_count=session.slider_count, tfidf_data=tfidf_data)
    st.table(dataframe)
