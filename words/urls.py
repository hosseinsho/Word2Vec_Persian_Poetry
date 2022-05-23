from django.urls import path
from words import views

urlpatterns = [
    path('vector/', views.words_vector),
    path('similarity/', views.words_similarity),
]
