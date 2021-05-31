from rest_framework import serializers
from .models import Kickrani
from django.db.models import Count

class KickraniSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','violation','image','datetime','helmet','location','person']


class DailyChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
<<<<<<< HEAD
        fields = ['kickId','brand','datetime']
=======
        fields = ['brand', 'num_brand']
>>>>>>> root_main

class AnnualChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','datetime']