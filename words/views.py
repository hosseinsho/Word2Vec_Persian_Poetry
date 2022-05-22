import numpy as np
from django.shortcuts import render
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances


def words_exists(word):
    embeddings = np.load('embeddings.npz')
    keys = embeddings.files
    if word in keys:
        return True


def get_most_similarity(word, weights=weights, tokenizer=tokenizer, n=15, type_='cosine'):
    if type_ is "euclidean":
        distance_matrix = euclidean_distances(weights)
    else:
        distance_matrix = cosine_distances(weights)
    word_to_sequences = tokenizer.texts_to_sequences([word])[0][0]
    index = np.argsort(distance_matrix[word_to_sequences])[:n]
    sequences_to_word = tokenizer.sequences_to_texts([index])[0]
    most_similarity = sequences_to_word.split(' ')
    return most_similarity

# Create your views here.
