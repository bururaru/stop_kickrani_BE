from django.urls import path
from . import api

urlpatterns = [
    path('', api.kickraniList, name="kickranis"),
    # path('detail/<str:pk>/', api.todoDetail, name="detail"),
    # path('create/', api.todoCreate, name="create"),
    # path('update/<str:pk>/', api.todoUpdate, name='update'),
    # path('delete/<str:pk>/', api.todoDelete, name='delete'),
]