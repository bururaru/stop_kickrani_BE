import paho.mqtt.client as mqtt
import json
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from .detection import detect2
import torch

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
    if len(tmp) == 4:
        # 테스트 코드
        # cv2.rectangle(tmp[0], tmp[1], tmp[2], (255,0,0), thickness=3, lineType=cv2.LINE_AA)
        # cv2.imshow('ImageWindow', tmp[0])
        # cv2.waitKey(10)
        #테스트 코드

        xyxy = torch.tensor([[float(tmp[1][0]), float(tmp[1][1]), float(tmp[2][0]), float(tmp[2][1])]])
        cropped = crop_one_box(xyxy, tmp[0])
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # cv2.imshow('ImageWindow', cropped)
        # cv2.waitKey(300)
        detect2(cropped, tmp[3])
        tmp = []

    if (msg.topic == "image") & (len(tmp) == 0):
        img = Image.open(BytesIO(base64.b64decode(msg.payload)))
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp.append(img)

    elif (msg.topic == "json") & (len(tmp) == 1):
        input_json = str(msg.payload.decode("utf-8"))
        # print(input_json)
        cord = json.loads(msg.payload)
        time = cord['time']
        cord = cord['cord']
        c1 = cord[0]
        c2 = cord[1]
        tmp.append(c1)
        tmp.append(c2)
        tmp.append(time)

def receive(request):
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

    # client.connect('52.79.80.91', 5556)
    client.connect('13.208.139.247', 1883)
    client.subscribe('image', 1)
    client.subscribe('json', 1)
    client.loop_forever()

def crop_one_box(xyxy, im, gain=1.02, pad=10, square=False):
    # Save an image crop as {file} with crop size multiplied by {gain} and padded by {pad} pixels
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), :: -1]
    return crop

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2