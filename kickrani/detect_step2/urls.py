from django.contrib import admin
from django.urls import path
from . import views, mqtt


urlpatterns = [
    path('main/', views.main),
    path('', mqtt.receive, name="mqtt"),
]