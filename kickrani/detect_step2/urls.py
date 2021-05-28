from django.contrib import admin
from django.urls import path
from . import views, detect_main, mqtt


urlpatterns = [
    path('main/', views.main),
    path('detect/', views.detect, name='detect'),
    path('detecting', detect_main.socket_receive, name='socket'),
    path('', mqtt.receive, name="mqtt")
]