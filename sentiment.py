from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import pandas as pd
import datetime
import spacy
import preprocessing as prep
import utils
import seaborn as sns
import matplotlib.pyplot as plt

def get_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_imdb_input_vector():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    x_train = prep.vectorize_sequences(train_data)
    x_test = prep.vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    return x_train, x_test, y_train, y_test

def train_imdb_sentiment_model(x_train, x_test, y_train, y_test, model):
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
    sent_model_eval = model.history
    print(sent_model_eval)
    return model


# Train sentiment model on Imdb dataset 
x_train, x_test, y_train, y_test = get_imdb_input_vector()
model = get_model()
model = train_imdb_sentiment_model(x_train, x_test, y_train, y_test, model)

# Read beer challenge csv
df = utils.read_beer_challenge_csv()

# Prepare Input vectors for beer reviews
word_index = imdb.get_word_index()
beer_x_data = df['review_text'][:5]
beer_y_data = df['review_overall'][:5]
print(beer_x_data)
print(beer_y_data)

beer_review_word_indexes, all_review_token_ids = prep.create_word_dict(beer_x_data)
beer_x_data_eval = prep.vectorize_sequences(all_review_token_ids)

# Predict sentiment score for the beer reviews
result = model.predict(beer_x_data_eval)

# Evaluate and compare predicted scores with actual score and find standard error
review_scores = np.array(beer_y_data)
std_error = (review_scores/5) - result
print("Review Overall Scores :: ", review_scores)
print("Predicted Scores :: ", result)
print("Standard Error between actual and predicted", std_error)

# plot the std. error
plt.plot(result*5, 'bo', label='predicted score')
plt.plot(review_scores, 'b', label='actual score')
plt.title( 'Standard Error - Model Score and Overall Score' )
plt.legend()
plt.show()


