from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Kickrani
from .serializer import KickraniSerializer
from .serializer import DailyChartSerializer
from .serializer import AnnualChartSerializer
from datetime import datetime
from django.utils.dateformat import DateFormat
import cv2

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

def kickraniDB(request,origin_frame):
    # print('!!',request)
    cv2.imwrite('image/'+str(1) + '.png', origin_frame)
    # print(str(1)+'.png'+' 파일이 저장되었습니다')
    #violation 1:2인이상, 2: 헬멧미착용, 3:2인이상 및 헬멧 미착용 4:
    if request["person"]>1:
        if request["person"]!=request["helmet"]:
            request["violation"] = 3
        else:
            request["violation"] = 1
    else:
        if request["helmet"]!=1:
            request["violation"] = 2
        # else:
        #     print("정상적인 사용자 입니다")
        #     return Response(serializer.data)
    serializer = KickraniSerializer(data=request) #data=request.data
    if(serializer.is_valid()):
        print('DB 저장 완료')
        # print('!!!!!!!!!!!!!!!!!!!!db접속!!!!!!!!!!!!!!!!!!!!!!', request, type(request))
        serializer.save()
    else:
        print('false')
    return Response(serializer.data)
