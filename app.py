import streamlit as st
import pickle
import pandas as pd
import requests
import re
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import os
import gdown



# Helpers: title cleaning

def clean_title(title: str) -> str:
    return re.sub(r'[^\w\s]', '', title).strip()


# OMDb: title-based details

def get_movie_details_omdb(title: str) -> dict:
    api_key = "aaec11cf"  # replace with your OMDb API key
    q = clean_title(title)
    url = f"http://www.omdbapi.com/?t={q}&apikey={api_key}"
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
    except Exception:
        data = {"Response": "False"}

    if data.get("Response") == "True":
        return {
            "title": data.get("Title", title),
            "poster": data.get("Poster", "https://via.placeholder.com/500x750?text=No+Poster"),
            "rating": data.get("imdbRating", "N/A"),
            "genre": data.get("Genre", "N/A"),
            "year": data.get("Year", "N/A"),
        }
    else:
        return {
            "title": title,
            "poster": "https://via.placeholder.com/500x750?text=No+Poster",
            "rating": "N/A",
            "genre": "N/A",
            "year": "N/A",
        }


# IMDb poster fallback

def get_poster_imdb(title: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.imdb.com/find?q={requests.utils.quote(title)}&s=tt"
        r = requests.get(search_url, headers=headers, timeout=8)
        if r.status_code != 200:
            return "https://via.placeholder.com/500x750?text=No+Poster"

        soup = BeautifulSoup(r.content, "html.parser")
        a = soup.find("a", href=re.compile(r"/title/tt\d+"))
        if not a or "href" not in a.attrs:
            return "https://via.placeholder.com/500x750?text=No+Poster"

        m = re.search(r"tt\d+", a["href"])
        if not m:
            return "https://via.placeholder.com/500x750?text=No+Poster"
        movie_id = m.group()

        title_url = f"https://www.imdb.com/title/{movie_id}/"
        rt = requests.get(title_url, headers=headers, timeout=8)
        if rt.status_code != 200:
            return "https://via.placeholder.com/500x750?text=No+Poster"

        tsoup = BeautifulSoup(rt.content, "html.parser")
        img = tsoup.find("img", alt=re.compile(r"Poster", re.IGNORECASE))
        if img and img.get("src"):
            poster_url = img["src"]
            poster_url = re.sub(r'@.*\.jpg', '@SY500.jpg', poster_url)
            return poster_url
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"


# Unified details with poster fallback

def get_movie_details(title: str) -> dict:
    details = get_movie_details_omdb(title)
    if "No+Poster" in details["poster"]:
        imdb_poster = get_poster_imdb(title)
        if "No+Poster" not in imdb_poster:
            details["poster"] = imdb_poster
    return details


# Data loading

movies_list = pickle.load(open("movies.pkl", "rb"))
movies = pd.DataFrame(movies_list)

file_id = "1elaa7DTiDRtjrjz-SpOGbY2Ftdz-E7D0"
url = f"https://drive.google.com/uc?id={file_id}"
output = "similarity.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


with open(output, "rb") as f:
    similarity = pickle.load(f)


# UI

st.title("🎬 Movie Recommender System")
selected_movie_name = st.selectbox("Choose Your Movie", movies["title"].values)

# Number input for recommendations
num_recommendations = st.number_input(
    "How many recommendations would you like?",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)


# Recommendation logic

def recommend(movie: str) -> list[str]:
    movie_lower = movie.lower().strip()
    exact_match = movies[movies["title"].str.lower() == movie_lower]

    if len(exact_match) > 0:
        movie_index = exact_match.index[0]
    else:
        titles = movies["title"].values
        best_match, best_ratio = None, 0.6
        for t in titles:
            ratio = SequenceMatcher(None, movie_lower, t.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_match = ratio, t
        if best_match is None:
            st.error(f"Movie '{movie}' not found. Try a different title.")
            return []
        movie_index = movies[movies["title"] == best_match].index[0]

    distances = similarity[movie_index]
    top_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
    return [movies.iloc[i[0]].title for i in top_indices]


# Display recommendations

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    if recommendations:
        # Display in rows of 5
        for row_start in range(0, len(recommendations), 5):
            row_movies = recommendations[row_start:row_start+5]
            cols = st.columns(len(row_movies))
            for idx, movie in enumerate(row_movies):
                with cols[idx]:
                    details = get_movie_details(movie)
                    poster = details.get("poster")
                    if poster and poster != "N/A":
                        st.image(poster,width=170)
                    else:
                        st.image("https://via.placeholder.com/170x250?text=No+Poster", width=170)
                    st.markdown(f"**{details['title']}**")
                    st.markdown(f"Rating: {details['rating']}⭐")
                    st.markdown(f"Genre: {details['genre']}")
                    st.markdown(f"Year: {details['year']}")
