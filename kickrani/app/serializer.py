from rest_framework import serializers
from .models import Rider
from .models import Image
from .models import Rider_information
from django.db.models import Count

class ImageSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Image
        fields = ['image_ID','datetime','location','rider_number']

class RiderSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Rider
        fields = ['rider_ID','image_ID','rider_location','rider_percentage','brand','helmet_number','person_number','datetime','foul','num_brand','location']


class InformationSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Rider_information
        fields = ['helmet_location','helmet_percentage','person_location','person_percentage']

class DailyChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Rider
        fields = ['brand', 'num_brand']

class AnnualChartSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Rider
        fields = ['rider_ID','brand','datetime']