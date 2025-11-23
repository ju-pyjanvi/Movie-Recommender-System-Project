#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd


# In[33]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[34]:


movies.head(1)


# In[35]:


credits.head(1)


# In[36]:


movies = movies.merge(credits,on='title')


# In[37]:


movies.head(1)


# In[38]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[39]:


movies.head()


# In[40]:


movies.isnull().sum()


# In[41]:


movies.dropna(inplace=True)


# In[42]:


movies.duplicated().sum()


# In[43]:


movies.iloc[0].genres


# In[44]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[45]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[46]:


movies['genres'] = movies['genres'].apply(convert)


# In[47]:


movies['keywords']=movies['keywords'].apply(convert)


# In[48]:


movies.head()


# In[49]:


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


# In[50]:


movies['cast'] = movies['cast'].apply(convert3)


# In[51]:


movies.head()


# In[52]:


movies['crew'][0]


# In[53]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[54]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[55]:


movies.head()


# In[56]:


movies['overview'][0]


# In[57]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[58]:


movies.head()


# In[59]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[60]:


movies.head()


# In[61]:


movies['tags'] = movies['overview'] + movies['cast'] + movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords']


# In[62]:


movies.head()


# In[63]:


new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[64]:


new_df.head()


# In[65]:


new_df['tags'][0]


# In[66]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[67]:


new_df.head()


# In[68]:


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[69]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[70]:


vectors[0]


# In[71]:


cv.get_feature_names_out()


# In[72]:


import nltk
from nltk.stem.porter import PorterStemmer
from difflib import SequenceMatcher
ps = PorterStemmer()


# In[73]:


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[74]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[75]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[90]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:11]


# In[92]:


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


# In[78]:


new_df.head()


# In[85]:


import pickle


# In[87]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[88]:


new_df['title'].values


# In[93]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




