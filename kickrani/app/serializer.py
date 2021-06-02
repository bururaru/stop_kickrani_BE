from rest_framework import serializers
from .models import Kickrani
from .models import Rider
from .models import Violation
from django.db.models import Count

class KickraniSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','violation','image','datetime','helmet','location','person']


class DailyChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['brand', 'num_brand']

class AnnualChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Kickrani
        fields = ['kickId','brand','datetime']

class RiderSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Rider
        fields = ['riderLocation','riderPercentage']

class ViolationSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Violation
        fields = ['helmetLocation','personLocation','personPercentage']

