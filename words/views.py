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


def get_most_similarity(word, weights, tokenizer, n=15, type_='cosine'):
    if type_ == "euclidean":
        distance_matrix = euclidean_distances(weights)
    else:
        distance_matrix = cosine_distances(weights)
    word_to_sequences = tokenizer.texts_to_sequences([word])[0][0]
    index = np.argsort(distance_matrix[word_to_sequences])[:n]
    sequences_to_word = tokenizer.sequences_to_texts([index])[0]
    most_similarity = sequences_to_word.split(' ')
    return most_similarity


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

        with open("templates/vector_pages/post.html",'w') as f:
            f.write(html)
        context = {
           'word': word,
           "vector": embeddings[word]
        }
        return render(request, 'vector_pages/post.html', context=context)

