#necessary imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


usecols = ["MAL_ID", "Name", "Score", "Genres", "Type", "Episodes", "Premiered",
           "Studios", "Source", "Rating", "Members"] #naming the columns

anime_data=pd.read_csv('anime.csv',usecols=usecols) #load anime data set through anime.csv

# print("anime_data.shape:", anime_data.shape) #print the shape of the data set

anime_data.head(10) #print the top 10  data from anime_data dataframe

anime_data= anime_data.drop(['MAL_ID','Members', 'Premiered'], axis=1)

# removing unknown datas from the dataframe
def process_multilabel(series):
    series = series.split(",")
    if "Unknown" in series:
        series.remove("Unknown")
    return series

anime_data["Genres"] = anime_data["Genres"].map(process_multilabel)
anime_data["Studios"] = anime_data["Studios"].map(process_multilabel)
anime_data["Score"] = anime_data["Score"].replace("Unknown", 0).astype(float)
anime_data["Episodes"] = anime_data["Episodes"].replace("Unknown", 0).astype(int)

anime_data.head() #print the top 5 data from anime_data dataframe


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')


# Filling NaNs with empty string
genres_original = anime_data['Genres'].fillna('').astype(str)
genres_vector_tf_idf = tfv.fit_transform(genres_original)

# print("genres_vector_tf_idf.shape:", genres_vector_tf_idf.shape) #print the shape of the genres_vector


def get_recommended(vector, query_index, n_neighbors=10):
    model_knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors)
    model_knn.fit(csr_matrix(vector))
    distances, indices = model_knn.kneighbors(vector[query_index,:].reshape(1, -1), n_neighbors = n_neighbors)#finding the distances between a query and all the examples in the data
    result = []
    for i in range(0, len(distances.flatten())):
        index = indices.flatten()[i]
        if index == query_index:
            continue
        result.append(anime_data.iloc[index])   
    return pd.DataFrame(result)


def user_input(use_input):
    try:
        idx = anime_data['Name'][anime_data['Name'] == use_input].index[0]
        print(get_recommended(genres_vector_tf_idf, idx, 10))
    except:
        print("Please enter a valid anime name for more help try using")
        print(anime_data['Name'].head())

x = input("Enter an anime name: ")
user_input(x)