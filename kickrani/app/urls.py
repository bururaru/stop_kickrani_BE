from django.urls import path
from . import api

urlpatterns = [
    path('', api.kickraniList, name="kickranis"),
    path('dailychart/', api.dailyChart, name="dailychart"),
    path('annualchart/', api.annualChart, name="dailychart"),
    path('create/', api.kickraniCreate, name="create"),
    path('db/', api.kickraniDB, name="db"),
    # path('update/<str:pk>/', api.todoUpdate, name='update'),
    # path('delete/<str:pk>/', api.todoDelete, name='delete'),
]