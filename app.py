from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)



def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv",encoding="latin-1")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    df = movies.merge(credits, on="title")

    df = df[[
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "image_url"   
    ]]

    return df


def extract_names(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except:
        return []


def extract_director(text):
    try:
        return [i["name"] for i in ast.literal_eval(text) if i["job"] == "Director"]
    except:
        return []


def clean_list(lst):
    return [i.replace(" ", "") for i in lst]


def preprocess(df):
    df["overview"] = df["overview"].fillna("")

    df["genres"] = df["genres"].apply(extract_names).apply(clean_list)
    df["keywords"] = df["keywords"].apply(extract_names).apply(clean_list)
    df["cast"] = df["cast"].apply(lambda x: extract_names(x)[:5]).apply(clean_list)
    df["crew"] = df["crew"].apply(extract_director).apply(clean_list)

    def create_tags(row):
        return (
            " ".join(row["genres"]) + " " +
            " ".join(row["keywords"]) + " " +
            " ".join(row["cast"]) + " " +
            " ".join(row["crew"]) + " " +
            row["overview"]
        )

    df["tags"] = df.apply(create_tags, axis=1)
    df["tags"] = df["tags"].apply(lambda x: x.lower())

    return df


df = preprocess(load_data())
df = df.reset_index(drop=True)



tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["tags"])

similarity = cosine_similarity(tfidf_matrix)




def recommend(movie_name, top_n=6):
    movie_name = movie_name.lower()

    matches = df[df["title"].str.lower() == movie_name]

    if matches.empty:
        return []

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    movies = df.iloc[movie_indices][["title", "image_url"]]

    results = []
    for _, row in movies.iterrows():
        results.append({
            "title": row["title"],
            "poster": row["image_url"] if pd.notna(row["image_url"]) else "/static/default.jpg"
        })

    return results




@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("q", "").lower()

    if not query:
        return jsonify({"suggestions": []})

    results = df[df["title"].str.lower().str.contains(query)].head(6)

    suggestions = []
    for _, row in results.iterrows():
        suggestions.append({
            "title": row["title"],
            "poster": row["image_url"] if pd.notna(row["image_url"]) else "/static/default.jpg"
        })

    return jsonify({"suggestions": suggestions})




@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        movie = request.form.get("movie")
        if movie:
            recommendations = recommend(movie)

    return render_template("index.html", recommendations=recommendations)




if __name__ == "__main__":
    app.run(debug=True)