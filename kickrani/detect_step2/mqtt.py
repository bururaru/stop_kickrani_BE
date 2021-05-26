import paho.mqtt.client as mqtt
import json
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK!!!")
    else:
        print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    global tmp
    if len(tmp) == 3:
        cv2.rectangle(tmp[0], tmp[1], tmp[2], (255,0,0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imshow('ImageWindow', tmp[0])
        cv2.waitKey(500)
        tmp = []

    if msg.topic == "image":
        print("이미지 검출됨")
        img = Image.open(BytesIO(base64.b64decode(msg.payload)))
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp.append(img)
        # cv2.imshow('ImageWindow', img)
        # cv2.waitKey(500)
        # detect2(frame)

    elif msg.topic == "json":
        # input_json = str(msg.payload.decode("utf-8"))
        cord = json.loads(msg.payload)
        cord = cord['cord']
        c1 = cord[0]
        c2 = cord[1]
        tmp.append(c1)
        tmp.append(c2)
        
global tmp
tmp = []
# def receive_mqtt():
# 새로운 클라이언트 생성
client = mqtt.Client()
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_subscribe(topic 구독),
# on_message(발행된 메세지가 들어왔을 때)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_subscribe = on_subscribe
client.on_message = on_message
# address : localhost, port: 1883 에 연결
client.connect('localhost', 1883)
# common topic 으로 메세지 발행
client.subscribe('image', 1)
client.subscribe('json', 1)
client.loop_forever()