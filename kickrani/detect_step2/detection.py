import json
import time
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from .loadimage import *
from .apps import DetectStep2Config
import sys
from app.api import imageDB
from app.api import riderDB
from app.api import informationDB
# mqtt 통신에서 img를 인풋으로 받아 실행되는 detection
def detect2(frame, frame_loc, frame_prob, c_time, origin_frame):
    # print("detect 시작")
    # Initialize
    project = './detect_step2/detect_result'
    name = 'result_img'
    source = frame
    save_img = True
    view_img = False
    save_txt = False
    augment = False
    imgsz = 352
    conf_thres = 0.75
    iou_thres = 0.7
    line_thickness=3
    hide_conf = False
    classes = None
    save_conf = True
    agnostic_nms = False
    save_crop = False
    hide_labels = False
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=True)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = DetectStep2Config.device
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    # apps.py에서 모델 import
    model1 = DetectStep2Config.model1
    model2 = DetectStep2Config.model2
    model3 = DetectStep2Config.model3
    stride1 = int(model1.stride.max())  # model stride
    stride2 = int(model2.stride.max())  # model stride
    stride3 = int(model3.stride.max())  # model stride
    imgsz1 = check_img_size(imgsz, s=stride1)  # check img_size
    imgsz2 = check_img_size(imgsz, s=stride2)  # check img_size
    imgsz3 = check_img_size(imgsz, s=stride3)  # check img_size
    names1 = model1.module.names if hasattr(model1, 'module') else model1.names  # get class names
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names  # get class names
    names3 = model3.module.names if hasattr(model3, 'module') else model3.names  # get class names
    if half:
        model1.half()
        model2.half()
        model3.half()
    classify = False
    # Set Dataloader
    t0 = time.time()
    vid_path, vid_writer = None, None
    dataset = custom_load(source)
    # print(len(dataset))
    # print((dataset[0]))
    # print((dataset[1]))
    # print((dataset[2]))
    # print((dataset[3]))
    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    path = dataset[0]
    img = dataset[1]
    im0s = dataset[2]
    vid_cap = dataset[3]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    pred1 = model1(img, augment=augment)[0]
    pred2 = model2(img, augment=augment)[0]
    pred3 = model3(img, augment=augment)[0]
    # Apply NMS
    pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes, agnostic_nms)
    pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes, agnostic_nms)
    pred3 = non_max_suppression(pred3, conf_thres, iou_thres, classes, agnostic_nms)
    t2 = time_synchronized()
    # Process detections kickboard
    kick_list = []
    kick_prob = []
    helmet_loc = []
    helmet_prob = []
    person_loc = []
    person_prob = []
    for i, det in enumerate(pred1):  # detections per image
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        pp = p.name[:-4] + '_1.jpg'
        save_path = str(save_dir / pp)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for opt.save_crop
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names1[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names1[c] if hide_conf else f'{names1[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=[255,255,0], line_thickness=line_thickness)
            kick_list.append(names1[c])
            kick_prob.append(float(conf))
        # Print time (inference + NMS)
        imc = cv2.resize(im0, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_AREA)
        cv2.imshow('ImageWindow', imc)
        cv2.waitKey(200)
        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)
    # Process detections helmet
    n2 = 0
    for i, det in enumerate(pred2):  # detections per image
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        pp = p.name[:-4] + '_2.jpg'
        save_path = str(save_dir / pp)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for opt.save_crop
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                ishelmet = False
                if int(c) == 0:
                    ishelmet = True
                    n2 = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n2} {names2[int(c)]}{'s' * (n2 > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names2[c] if hide_conf else f'{names2[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=[255,0,0], line_thickness=line_thickness)
                if ishelmet:
                    helmet_loc.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    helmet_prob.append(float(conf))
                    imc = cv2.resize(im0, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_AREA)
                    cv2.imshow('ImageWindow', imc)
                    cv2.waitKey(200)
        num_helmet = int(n2)
        if len(helmet_loc) == 0:
            helmet_loc.append(0)
            helmet_prob = 0

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)
    n3 = 0
    # Process detections person
    for i, det in enumerate(pred3):  # detections per image
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        pp = p.name[:-4] + '_3.jpg'
        save_path = str(save_dir / pp)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for opt.save_crop
        n3 = 0
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n3 = (det[:, -1] == c).sum()  # detections per class
                s += f"{n3} {names3[int(c)]}{'s' * (n3 > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names3[c] if hide_conf else f'{names3[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=[0, 0, 255], line_thickness=line_thickness)
                person_loc.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                person_prob.append(float(conf))
                imc = cv2.resize(im0, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_AREA)
                cv2.imshow('ImageWindow', imc)
                cv2.waitKey(200)
        num_person = int(n3)

    print(f'Done. ({time.time() - t0:.3f}s)')
    # print("detect 끝")
    # if num_person:
    print(f'사람 수 : {num_person}')
    # elif num_helmet:
    print(f'헬멧 수 : {num_helmet}')
    # else:
    #     print("사람, 헬멧 탐지 안됨!")
    # 정확도가 가장 높게 나온 킥보드 브랜드 하나만 반환
    if kick_list:
    # if True:
        kick_list = kick_list[kick_prob.index(max(kick_prob))]
        # kick_list=kick_list[0]
        print(f'킥보드 브랜드 : {kick_list}')
        # 헬멧 수보다 사람 수가 많으면 위반 ( 사람수가 3 이상으로 많은 숫자로 탐지되어도 동일하게 적용)
        if num_helmet < num_person:
            print("위반!!!!!!!!!!!!!!!!!!!!!")
            py_data1 = {'datetime': c_time, "location": "강남역", "rider_number": num_person}  # Json 형태로 변환
            py_data2 = {"rider_location": str(frame_loc), "rider_percentage": frame_prob, 'brand': str(kick_list),
                        "helmet_number": num_helmet, "person_number": num_person,'datetime': c_time}
            # py_data2={"helmet_number": num_helmet,"person_number": num_person}
            #'person_location': '[[24, 10, 172, 411]]', 'person_percentage': [0.7623780369758606]
            py_data3={"helmet_location": str(helmet_loc), "helmet_percentage": float(helmet_prob),"person_location": str(person_loc), "person_percentage": float(person_prob)}
            imageDB(py_data1, origin_frame)
            riderDB(py_data2,py_data3)
            # informationDB(py_data3)
    else:
        print("위반 아님")