# import argparse
# import time
# from pathlib import Path
#
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
#
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized
#
#
# def detect(opt):
#     #Initialize
#     project = 'detect_result'
#     name = 'result_img'
#     source = './test_img'
#     save_img = True
#
#     # weights1 = 'kickboard.pt'
#     # weights2 = 'helmet.pt'
#     # weights3 = 'person.pt'
#
#     view_img, save_txt, imgsz = opt.view_img, opt.save_txt, opt.img_size
#     # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
#     # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#     #     ('rtsp://', 'rtmp://', 'http://', 'https://'))
#
#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=True)  # increment run
#     # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
#
#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#
#     # Load model
#     # model1 = attempt_load(weights1, map_location=device)  # load FP32 model
#     # model2 = attempt_load(weights2, map_location=device)  # load FP32 model
#     # model3 = attempt_load(weights3, map_location=device)  # load FP32 model
#     stride1 = int(model1.stride.max())  # model stride
#     stride2 = int(model2.stride.max())  # model stride
#     stride3 = int(model3.stride.max())  # model stride
#     imgsz1 = check_img_size(imgsz, s=stride1)  # check img_size
#     imgsz2 = check_img_size(imgsz, s=stride2)  # check img_size
#     imgsz3 = check_img_size(imgsz, s=stride3)  # check img_size
#     names1 = model1.module.names if hasattr(model1, 'module') else model1.names  # get class names
#     names2 = model2.module.names if hasattr(model2, 'module') else model2.names  # get class names
#     names3 = model2.module.names if hasattr(model3, 'module') else model2.names  # get class names
#     # if half:
#     #     model.half()  # to FP16
#
#     # Second-stage classifier
#     classify = False
#     # if classify:
#     #     modelc = load_classifier(name='resnet101', n=2)  # initialize
#     #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
#
#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     dataset = LoadImages(source, img_size=imgsz, stride=stride1)
#     # 이부분에 stride 사이즈로 인한 문제 발생 가능성 높음
#
#     # Run inference
#     # if device.type != 'cpu':
#     #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Inference
#         t1 = time_synchronized()
#         pred1 = model1(img, augment=opt.augment)[0]
#         pred2 = model2(img, augment=opt.augment)[0]
#         pred3 = model3(img, augment=opt.augment)[0]
#
#         # Apply NMS
#         pred1 = non_max_suppression(pred1, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
#                                    max_det=opt.max_det)
#         pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
#                                    max_det=opt.max_det)
#         pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
#                                    max_det=opt.max_det)
#         t2 = time_synchronized()
#
#         # Apply Classifier
#         # if classify:
#         #     pred = apply_classifier(pred, modelc, img, im0s)
#
#         # Process detections 1
#         for i, det in enumerate(pred1):  # detections per image
#             p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
#             p = Path(p)  # to Path
#             pp = p.name[:-4] + '_1.jpg'
#             save_path = str(save_dir / pp)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names1[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                     if save_img or opt.save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if opt.hide_labels else (names1[c] if opt.hide_conf else f'{names1[c]} {conf:.2f}')
#                         plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
#                         if opt.save_crop:
#                             save_one_box(xyxy, imc, file=save_dir / 'crops' / names1[c] / f'{p.stem}.jpg', BGR=True)
#
#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')
#
#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(im0)
#
#         # Process detections 2
#         for i, det in enumerate(pred2):  # detections per image
#             p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
#             p = Path(p)  # to Path
#             pp = p.name[:-4] + '_2.jpg'
#             save_path = str(save_dir / pp)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                     if save_img or opt.save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if opt.hide_labels else (names2[c] if opt.hide_conf else f'{names2[c]} {conf:.2f}')
#                         plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
#                         if opt.save_crop:
#                             save_one_box(xyxy, imc, file=save_dir / 'crops' / names2[c] / f'{p.stem}.jpg', BGR=True)
#
#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')
#
#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(im0)
#
#         # Process detections 3
#         for i, det in enumerate(pred3):  # detections per image
#             p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
#             p = Path(p)  # to Path
#             pp = p.name[:-4] + '_3.jpg'
#             save_path = str(save_dir / pp)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names3[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                     if save_img or opt.save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if opt.hide_labels else (names3[c] if opt.hide_conf else f'{names3[c]} {conf:.2f}')
#                         plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
#                         if opt.save_crop:
#                             save_one_box(xyxy, imc, file=save_dir / 'crops' / names3[c] / f'{p.stem}.jpg', BGR=True)
#
#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')
#
#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(im0)
#
#     # if save_txt or save_img:
#     #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#     #     print(f"Results saved to {save_dir}{s}")
#
#     print(f'Done. ({time.time() - t0:.3f}s)')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     opt = parser.parse_args()
#     print(opt)
#     check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
#
#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
#                 detect(opt=opt)
#                 strip_optimizer(opt.weights)
#         else:
#             detect(opt=opt)
