from django.urls import path
from . import api

urlpatterns = [
    path('', api.riderList, name="riderlist"),
    path('dailychart/', api.dailyChart, name="dailychart"),
    path('annualchart/', api.annualChart, name="dailychart"),
    path('imagecreate/', api.imageCreate, name="imagecreate"),
    path('ridercreate/', api.riderCreate, name="ridercreate"),
    path('informationcreate/', api.InformationCreate, name="informationcreate"),
    path('db/', api.riderDB, name="db"),
    # path('update/<str:pk>/', api.todoUpdate, name='update'),
    # path('delete/<str:pk>/', api.todoDelete, name='delete'),
]