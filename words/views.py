import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

embedding_dir = 'words/embeddings.npz'


def words_exists(word):
    embeddings = np.load(embedding_dir)
    keys = embeddings.files
    if word in keys:
        return True


def get_most_similarity(word, weights, n=15, type_='cosine'):
    tokenizer = 0
    with open('words/tokenizer', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    if type_ == "euclidean":
        distance_matrix = euclidean_distances(weights)
    else:
        distance_matrix = cosine_distances(weights)
    word_to_sequences = tokenizer.texts_to_sequences([word])[0][0]
    index = np.argsort(distance_matrix[word_to_sequences])[:n]
    sequences_to_word = tokenizer.sequences_to_texts([index])[0]
    most_similarity = sequences_to_word.split(' ')
    return most_similarity


def distance_two_words(word1, word2, type_='cosine'):
    if type_ == "euclidean":
        dist = np.linalg.norm(word2 - word1)
    else:
        dist = 1 - np.dot(word1, word2) / (np.linalg.norm(word1) * np.linalg.norm(word2))
    dist = 0 if dist < 0.000001 else dist
    return dist


def words_vector(request):
    if request.method == 'GET':
        return render(request, 'vector_pages/get.html')

    elif request.method == 'POST':
        word = request.POST.get("word", None)
        if not word:
            context = {"status": 400, "message": "لطفا کلمه ای وارد کنید"}
            return render(request, 'vector_pages/get.html', context=context)
        if not words_exists(word):
            context = {"status": 404, "message": "کلمه یافت نشد"}
            return render(request, 'vector_pages/get.html', context=context)
        embeddings = np.load(embedding_dir)

        df = pd.DataFrame(embeddings[word])
        html = df.to_html()

        with open("templates/vector_pages/post.html", 'w') as f:
            f.write(html)
        context = {
            'word': word,
            "vector": embeddings[word]
        }
        return render(request, 'vector_pages/post.html', context=context)


def words_similarity(request):
    if request.method == 'GET':
        return render(request, 'similarity/get.html')

    elif request.method == 'POST':
        word = request.POST.get("word", None)
        if not word:
            context = {"status": 400, "message": "لطفا کلمه ای وارد کنید"}
            return render(request, 'similarity/get.html', context=context)
        if not words_exists(word):
            context = {"status": 404, "message": "کلمه یافت نشد"}
            return render(request, 'similarity/get.html', context=context)
        embeddings = np.load(embedding_dir)
        arrays = []
        for item in embeddings.files:
            arrays.append(embeddings[item])
        weights = np.vstack(arrays)
        similar_words = []
        for sim_word in get_most_similarity(word, weights, type_="cosine"):
            similar_words.append(
                (sim_word, distance_two_words(embeddings[word], embeddings[sim_word], type_="cosine"))
            )
        print(similar_words)

        context = {
            'word': word,
            "similar_words": similar_words
        }
        return render(request, 'similarity/post.html', context=context)
