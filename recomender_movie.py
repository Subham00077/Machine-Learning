import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



movies=pd.read_csv('movies.csv')
movies.dropna(inplace=True)
movies = movies.head(1000) 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)




tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])




cosine_sim = cosine_similarity(tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()




def recommend_movie(title, num_recommendations=5):
    if title not in indices:
        return "Movie not found in dataset."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()




print("🎥 Recommendations for 'Toy Story (1995)':")
recommendations = recommend_movie('Toy Story (1995)', 5)
print(recommendations)
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")


