import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import preprocessing as prep
import utils

def similarity_spacy(review1, review2):
    doc = nlp(review1)
    doc1 = nlp(review2)
    return doc.similarity(doc1)

def create_similar_groups(similarity_mat, profile_names):
    pass

def apply_cosine_similarity_technique():
    beer_reviews = df['review_text'][:20]
    profile_names = df['review_profileName'][:20]
    beer_review_word_indexes, all_review_token_ids = prep.create_word_dict(beer_reviews)
    X = prep.vectorize_sequences(all_review_token_ids)
    similarity_mat = cosine_similarity(X, Y=None)
    print("Cosine Similarity matrix ::", similarity_mat)
    #print(profile_names)
    #create_similar_groups(similarity_mat, profile_names)

def plot_elbow_method(K, Sum_of_squared_distances):
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def create_input_vector(beer_reviews):
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(beer_reviews)
    return X

def apply_K_means(X, range_min, range_max):
    Sum_of_squared_distances = []
    K = range(range_min, range_max)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    return K, Sum_of_squared_distances

def do_clustering_beer_reviews(X, true_k, profile_names):
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels=model.labels_
    beer_review_cl=pd.DataFrame(list(zip(profile_names,labels)),columns=['user','cluster']) 
   
    return beer_review_cl.sort_values(by=['cluster'])

def apply_clustering_technique(df):
    beer_reviews = df['review_text'][:1000]
    profile_names = df['review_profileName'][:1000]

    X = create_input_vector(beer_reviews)
    
    # Repeat steps 1 and 2 until find optimal K value
    # Step 1 - select range and find K and sum of squared distances
    range_min = 8
    range_max = 20
    K, Sum_of_squared_distances = apply_K_means(X, range_min, range_max)
    # Step 2 - observe the plot and select optimal k
    plot_elbow_method(K, Sum_of_squared_distances)

    true_k = 19   # selected based on 100K samples
    return do_clustering_beer_reviews(X,true_k, profile_names)


df = utils.read_beer_challenge_csv()
df = utils.drop_na_rows(df)

# Apply clustering technique
beer_review_clusters = apply_clustering_technique(df)
print(beer_review_clusters)

# Apply cosing similarity technique
apply_cosine_similarity_technique()
