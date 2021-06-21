from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Rider
from .models import Image
from .models import Rider_information
from .serializer import DailyChartSerializer
from .serializer import AnnualChartSerializer
from .serializer import ImageSerializer
from .serializer import RiderSerializer
from .serializer import InformationSerializer
from datetime import datetime
from django.utils.dateformat import DateFormat
import cv2
from django.db.models import Count
import boto3
import json

with open('./secrets.json')as json_file:
    json_data = json.load(json_file)

aws = json_data["AWS"]

@api_view(['GET'])
def riderList(request):
    riders = Rider.objects.all()
    serializer = RiderSerializer(riders, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def dailyChart(request):
    # now = DateFormat(datetime.now()).format('Ymd')
    # now=now[:4]+'-'+now[4:6]+'-'+now[6:8] #2021-05-21 형태로 만들기 위한 코드
    # chart = Kickrani.objects.filter(brand__contains=now)
    chart = Rider.objects.values('brand').annotate(num_brand=Count('brand')).order_by('brand')
    serializer = DailyChartSerializer(chart, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def annualChart(request):
    now = DateFormat(datetime.now()).format('Ymd')
    now=now[:4] #2021-05-21 형태로 만들기 위한 코드
    print(now)
    chart = Rider.objects.filter(datetime__contains=now)
    serializer = AnnualChartSerializer(chart, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def imageCreate(request):
    serializer = ImageSerializer(data=request.data)
    if(serializer.is_valid()):
        serializer.save()
    return Response(serializer.data)

@api_view(['POST'])
def riderCreate(request):
    serializer = RiderSerializer(data=request.data)
    if(serializer.is_valid()):
        serializer.save()
    return Response(serializer.data)

@api_view(['POST'])
def InformationCreate(request):
    serializer = InformationSerializer(data=request.data)
    if(serializer.is_valid()):
        serializer.save()
    return Response(serializer.data)

#datetim형태 이미지저장명으로 바꾸
def dateHandler(date):
    date = date[:22].replace('-', '')
    date = date.replace(':', '')
    date = date.replace('.', '')
    date = date.replace(' ', '')
    return date


def riderDB(request,py_data3):
    request['location'] = "서울특별시 서초구 서초동 1374"  #장소 임의로 추가
    imageName = dateHandler(request["datetime"])
    request['image_ID'] = imageName

    #foul 1:2인이상, 2: 헬멧미착용, 3:2인이상 및 헬멧 미착용 4:
    if request["person_number"]>1:
        if request["person_number"]!=request["helmet_number"]:
            request["foul"] = 3
        else:
            request["foul"] = 1
    else:
        if request["helmet_number"]!=1:
            request["foul"] = 2
        # else:
        #     print("정상적인 사용자 입니다")
        #     return Response(serializer.data)

    serializer = RiderSerializer(data=request) #data=request.data
    print(serializer)
    if(serializer.is_valid()):
        print('table2 DB 저장 완료')
        serializer.save()
    else:
        print('table2 DB false')
    riderID=serializer["rider_ID"].value
    py_data3["rider_ID"]=riderID

    serializer1 = InformationSerializer(data=py_data3)
    print('###############',py_data3)
    print(serializer1)

    if (serializer1.is_valid()):
        print('table3 DB 저장 완료')
        serializer1.save()
    else:
        print('table3 DB false')

    return Response(serializer.data)

def imageDB(request, origin_frame):
    imageName = dateHandler(request["datetime"])
    cv2.imwrite('image/' + imageName + '.png', origin_frame)
    print(imageName + '.png' + ' 파일이 저장되었습니다')
    file_name = 'image/' + imageName + '.png'
    bucket = aws["bucket"]
    key = 'image/' + imageName + '.png'

    s3 = boto3.client(
        's3',
        aws_access_key_id=aws["aws_access_key_id"],
        aws_secret_access_key=aws["aws_secret_access_key"],
    )
    s3.upload_file(
        file_name,
        bucket,
        key,
        ExtraArgs={
            "ContentType": 'image/png',
        }
    )

    request['image_ID'] = imageName

    serializer = ImageSerializer(data=request)
    if(serializer.is_valid()):
        print('table1 DB 저장 완료')
        serializer.save()
    else:
        print('table1 DB false')

    return Response(serializer.data)


def informationDB(request):
    serializer = InformationSerializer(data=request)
    if(serializer.is_valid()):
        print('table3 DB 저장 완료')
        serializer.save()
    else:
        print('table3 DB false')
    return Response(serializer.data)