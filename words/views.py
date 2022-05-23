import pickle
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

        context = {
            'word': word,
            "similar_words": similar_words
        }
        return render(request, 'similarity/post.html', context=context)


def display_pca_scatterplot_3D(user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):
    embeddings = np.load(embedding_dir)
    arrays = []
    for item in embeddings.files:
        arrays.append(embeddings[item])
    weights = np.vstack(arrays)
    three_dim = PCA(random_state=0).fit_transform(weights)[:, :3]

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            z=three_dim[count:count + topn, 2],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)
        count = count + topn


    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


def display_tsne_scatterplot_3D(user_input=None, words=None, label=None, color_map=None, perplexity=0,
                                learning_rate=0, iteration=0, topn=5, sample=10):
    embeddings = np.load(embedding_dir)
    arrays = []
    for item in embeddings.files:
        arrays.append(embeddings[item])
    weights = np.vstack(arrays)
    three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                     n_iter=iteration).fit_transform(weights)[:, :3]

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []

    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0],
            y=three_dim[count:count + topn, 1],
            z=three_dim[count:count + topn, 2],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)
        count = count + topn

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


def pca_3d_view(request):
    if request.method == 'GET':
        embeddings = np.load(embedding_dir)
        display_pca_scatterplot_3D(['جان'], embeddings.files, topn=len(embeddings.files))
        return HttpResponse("<h1>map will load soon in new tap. please wait a moment</h1>")


def tsne_popular_words_3d_view(request):
    if request.method == 'GET':
        embeddings = np.load(embedding_dir)
        display_tsne_scatterplot_3D(['عقل', 'جان', 'چشم', 'سر', 'آب', 'نور', 'دست'], embeddings.files[:200]
                                    , perplexity=5, learning_rate=100, iteration=250)
        return HttpResponse("<h1>map will load soon in new tap. please wait a moment</h1>")

