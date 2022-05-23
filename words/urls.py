from django.urls import path
from words import views

urlpatterns = [
    path('vector/', views.words_vector),
    path('similarity/', views.words_similarity),
    path('similarity/3d/', views.similar15_map_words),
    path('3d/pca/', views.pca_3d_view),
    path('3d/tsne/', views.tsne_popular_words_3d_view),
]
