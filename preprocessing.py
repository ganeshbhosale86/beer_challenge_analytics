import spacy
import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_sm')
doc = nlp("This is test. This is another sentence.")

def create_word_dict(beer_reviews):
    count = 0
    word_indexes = {}
    all_review_word_ids = []
    stop_words = load_stopwords()
    for review in beer_reviews:
        doc = nlp(review)
        review_word_ids = []
        for token in doc:
            if token.lemma_ not in stop_words:
                if word_indexes.__contains__(token.lemma_):
                    review_word_ids.append(word_indexes.get(token.lemma_))
                    continue
                count = count + 1
                word_indexes[token.lemma_] = count
                review_word_ids.append(count)
        all_review_word_ids.append(review_word_ids)        
    return word_indexes, all_review_word_ids


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

def load_stopwords():
    stopwords = []
    f = open("resource/stopwords.txt","r")
    stopwords = f.read().split("\n")
    return stopwords



