from rest_framework import serializers
from .models import Kickrani

class KickraniSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','violation','image','datetime','helmet','location','person']


class DailyChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','datetime']

class AnnualChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','datetime']