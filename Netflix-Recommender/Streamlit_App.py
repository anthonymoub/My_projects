import pickle
import pandas as pd
import streamlit as st
from streamlit import session_state as session
from script import recommend_table
from script import get_movie_posters

@st.cache_data
def load_data():
    """
    load and cache data
    :return: tfidf data
    """
    tfidf_data = pd.read_csv("data/tfidf_data.csv")
    return tfidf_data


tfidf = load_data()

with open("data/titles.pkl", "rb") as f:
    movies = pickle.load(f)


dataframe = None

st.title("""
Netflix Recommendation System
This is a simple NLP based Content Based Recommender System. In short, Netflix movies/shows will be recommended to you based on the plot description of movies you input.
 """)

st.text("")
st.text("")
st.text("")
st.text("")

session.options = st.multiselect(label="Select Movies", options=movies)

st.text("")
st.text("")

session.slider_count = st.slider(label="Number of movies to recommend", min_value=5, max_value=20)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend")

st.text("")
st.text("")

if is_clicked:
    recommended_movies = recommend_table(tfidf[tfidf['title'].isin(session.options)].index.values, movie_count=session.slider_count, tfidf_data=tfidf)
    posters = get_movie_posters(recommended_movies , "a78a9010")
    if not posters:
        st.write("No movie posters found.")
    else:
        poster_html = ""
        for i, poster in enumerate(posters):
            if poster is not None:
                title = recommended_movies[i]
                poster_html += f'<div style="text-align: center;"><p>{title}</p><img src="{poster}" style="width: 200px; margin: 10px"/></div>'
        poster_div = f'<div id="poster-container" style="display: flex; overflow-x: auto; scroll-behavior: smooth; padding: 10px;">{poster_html}</div>'
        st.write(poster_div, unsafe_allow_html=True)


st.text("")
st.text("")
st.text("")
st.text("")

if dataframe is not None:
    st.table(dataframe)
