import streamlit as st
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_data
def load_data():
    def read_file(file):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except:
            return pd.read_csv(file, encoding="latin-1")

    movies = read_file("tmdb_5000_movies.csv")
    credits = read_file("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    return movies
df = load_data()
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def fetch_director(text):
    try:
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']
    except:
        return []
df["genres"] = df["genres"].apply(convert)
df["cast"] = df["cast"].apply(lambda x: convert(x)[:5])
df["crew"] = df["crew"].apply(fetch_director)

df["tags"] = df["genres"] + df["cast"] + df["crew"]
df["tags"] = df["tags"].apply(lambda x: " ".join(x))
tfidf = TfidfVectorizer(stop_words="english")
vectors = tfidf.fit_transform(df["tags"])

similarity = cosine_similarity(vectors)

st.title("🎬 Movie Recommender System")

st.write("Select your preferences:")

all_genres = sorted({g for sublist in df["genres"] for g in sublist})
all_actors = sorted({a for sublist in df["cast"] for a in sublist})
all_directors = sorted({d for sublist in df["crew"] for d in sublist})

selected_genres = st.multiselect("Select Genres", all_genres)
selected_actors = st.multiselect("Select Actors", all_actors)
selected_directors = st.multiselect("Select Directors", all_directors)
def recommend_by_preferences(genres, actors, directors, top_n=5):

    scores = []

    for i, row in df.iterrows():
        score = 0

        score += len(set(genres) & set(row["genres"]))
        score += len(set(actors) & set(row["cast"]))
        score += len(set(directors) & set(row["crew"]))

        scores.append(score)

    df["score"] = scores

    results = df[df["score"] > 0].sort_values("score", ascending=False)

    return results[["title", "score"]].head(top_n)
if st.button("Recommend Movies"):

    if not selected_genres and not selected_actors and not selected_directors:
        st.warning("Please select at least one option.")
    else:
        results = recommend_by_preferences(
            selected_genres,
            selected_actors,
            selected_directors
        )

        st.subheader("🎯 Recommendations")

        for i, row in results.iterrows():
            st.write(f"⭐ {row['title']} (Score: {row['score']})")