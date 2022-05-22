from django.urls import path
from words import views

urlpatterns = [
    path('vector/', views.words_vector),
]
