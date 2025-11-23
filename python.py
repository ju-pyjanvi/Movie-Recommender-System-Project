import numpy as np
import pandas as pd
import pickle

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)
credits.head(1)

movies = movies.merge(credits,on='title')
movies.head(1)

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()

movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()

movies.iloc[0].genres

import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

movies['genres'] = movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

movies.head()

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

movies.head()

movies['crew'][0]

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

movies.head()


movies['overview'][0]
movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


movies['tags'] = movies['overview'] + movies['cast'] + movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


new_df.head()

new_df['tags'][0]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')

vectors = cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names_out()

import nltk
from nltk.stem.porter import PorterStemmer
from difflib import SequenceMatcher
ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

from difflib import SequenceMatcher

def recommend(movie):
    movie_lower = movie.lower().strip()
    exact_match = new_df[new_df['title'].str.lower() == movie_lower]

    if len(exact_match) > 0:
        movie_index = exact_match.index[0]
    else:
        titles = new_df['title'].values
        best_match = None
        best_ratio = 0.6  

        for title in titles:
            ratio = SequenceMatcher(None, movie_lower, title.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = title

        if best_match is None:
            print(f"Movie '{movie}' not found. Try a different title.")
            return

        print(f"Found match: '{best_match}' (similarity: {best_ratio:.1%})")
        movie_index = new_df[new_df['title'] == best_match].index[0]

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nRecommendations for '{new_df.iloc[movie_index].title}':\n")
    for i in movies_list:
        print(f"  {new_df.iloc[i[0]].title}")



new_df.head()
pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))

