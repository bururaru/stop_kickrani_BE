from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Kickrani
from .serializer import KickraniSerializer
from .serializer import DailyChartSerializer
from .serializer import AnnualChartSerializer
from datetime import datetime
from django.utils.dateformat import DateFormat

@api_view(['GET'])
def kickraniList(request):
    kickranis = Kickrani.objects.all()
    serializer = KickraniSerializer(kickranis, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def dailyChart(request):
    now = DateFormat(datetime.now()).format('Ymd')
    now=now[:4]+'-'+now[4:6]+'-'+now[6:8] #2021-05-21 형태로 만들기 위한 코드
    chart = Kickrani.objects.filter(datetime__contains=now)
    serializer = DailyChartSerializer(chart, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def annualChart(request):
    now = DateFormat(datetime.now()).format('Ymd')
    now=now[:4] #2021-05-21 형태로 만들기 위한 코드
    print(now)
    chart = Kickrani.objects.filter(datetime__contains=now)
    serializer = AnnualChartSerializer(chart, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def kickraniCreate(request):
    serializer = KickraniSerializer(data=request.data)
    if(serializer.is_valid()):
        serializer.save()
    return Response(serializer.data)
