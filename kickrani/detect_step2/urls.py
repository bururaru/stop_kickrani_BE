from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('main/', views.main),
    path('detect/', views.detect, name='detect'),
]