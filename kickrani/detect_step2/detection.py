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
# sys.path.insert(0, "/app/api.py")
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from app.api import kickraniDB

# mqtt 통신에서 img를 인풋으로 받아 실행되는 detection

def detect2(frame, c_time):
    print("detect 시작")
    # Initialize
    project = './detect_step2/detect_result'
    name = 'result_img'
    origin_frame = frame
    #테스트용 코드
    # img_loc = './detect_step2/detect_before/maxresdefault.jpg'
    # socket_output = cv2.imread(img_loc)
    # source = socket_output
    source = frame
    save_img = True
    view_img = False
    save_txt = False
    augment = False
    imgsz = 352
    conf_thres = 0.25
    iou_thres = 0.45
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
    t0 = time.time()
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

    # Process detections 1
    kick_list = []
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
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names1[c] if hide_conf else f'{names1[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names1[c] / f'{p.stem}.jpg', BGR=True)
            kick_list.append(names1[c])
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)

    # Process detections 2
    for i, det in enumerate(pred2):  # detections per image
        det2 = det
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
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names2[c] if hide_conf else f'{names2[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names2[c] / f'{p.stem}.jpg', BGR=True)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)

    # Process detections 3
    for i, det in enumerate(pred3):  # detections per image
        det3 = det
        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        pp = p.name[:-4] + '_3.jpg'
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
                s += f"{n} {names3[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names3[c] if hide_conf else f'{names3[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names3[c] / f'{p.stem}.jpg', BGR=True)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("detect 끝")
    context = {'a': "킥라니 멈춰! detect 후"}

    py_data = {'brand': str(kick_list),'datetime': c_time,"helmet": len(det2), "person": len(det3)}  # Json 형태로 변환
    # origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    kickraniDB(py_data, origin_frame)
    # json_data = json.dumps(py_data, indent=4)  # Json 형태로 변환

