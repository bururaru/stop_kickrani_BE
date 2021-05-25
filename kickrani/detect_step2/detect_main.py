import socket
import cv2
import numpy as np
from django.shortcuts import render
from .detection import detect2

# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def socket_receive(request):
    #'ec2-3-36-96-187.ap-northeast-2.compute.amazonaws.com'
    HOST = 'ec2-52-79-80-91.ap-northeast-2.compute.amazonaws.com'
    PORT = 5556

    # TCP 사용
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    # 서버의 아이피와 포트번호 지정
    s.bind((HOST, PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)
    print('Socket now listening')

    # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    conn, addr = s.accept()
    i=0
    while True:
        # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')

        # data를 디코딩한다.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey(500)
        detect2(frame)
        # detect 추가 예정

    return render(request, 'main.html')
