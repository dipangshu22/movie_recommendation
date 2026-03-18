from flask import Flask, render_template, request
import pandas as pd
import ast

app = Flask(__name__)

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data():
    try:
        movies = pd.read_csv("tmdb_5000_movies.csv")
        credits = pd.read_csv("tmdb_5000_credits.csv")
    except:
        movies = pd.read_csv("tmdb_5000_movies.csv", encoding="latin-1")
        credits = pd.read_csv("tmdb_5000_credits.csv", encoding="latin-1")

    return movies.merge(credits, on="title")

df = load_data()

# ---------------------------
# PROCESS DATA
# ---------------------------
def extract_names(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def extract_director(text):
    try:
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']
    except:
        return []

df["genres"] = df["genres"].apply(extract_names)
df["cast"] = df["cast"].apply(lambda x: extract_names(x)[:5])
df["crew"] = df["crew"].apply(extract_director)

# ---------------------------
# RECOMMEND LOGIC
# ---------------------------
def recommend(genres, actors, directors, top_n=6):

    def score(row):
        return (
            len(set(genres) & set(row["genres"])) +
            len(set(actors) & set(row["cast"])) +
            len(set(directors) & set(row["crew"]))
        )

    df["score"] = df.apply(score, axis=1)

    results = df[df["score"] > 0].sort_values("score", ascending=False).head(top_n)

    return results[["title", "genres"]].to_dict(orient="records")

# ---------------------------
# ROUTE
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    all_genres = sorted({g for row in df["genres"] for g in row})
    all_actors = sorted({a for row in df["cast"] for a in row})
    all_directors = sorted({d for row in df["crew"] for d in row})

    recommendations = []

    if request.method == "POST":
        genres = request.form.getlist("genres")
        actors = request.form.getlist("actors")
        directors = request.form.getlist("directors")

        recommendations = recommend(genres, actors, directors)

    return render_template(
        "index.html",
        genres=all_genres,
        actors=all_actors,
        directors=all_directors,
        recommendations=recommendations
    )

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)