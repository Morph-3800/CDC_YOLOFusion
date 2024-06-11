# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images one by one.
"""

import argparse
import sys
from pathlib import Path

import cv2
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_suffix, \
    is_ascii, non_max_suppression, print_args, scale_coords, set_logging
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights='yolov5l.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):


    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # elif onnx:
    #     check_requirements(('onnx', 'onnxruntime'))
    #     import onnxruntime
    #     session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Run inference
    # if pt and device.type != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    if 'LLVIP' in source:
        source_file = source + '/infrared'
        img_file = os.listdir(source_file)

    elif 'peizhun' in source:
        source_file = source + '/lrir'
        img_file = os.listdir(source_file)
    else:
        img_file = os.listdir(source)

    # img_file = os.listdir(source)

    for file in img_file: 
        if 'Flir' in source: 
            if 'Preview' in file: 
                im0s_ther = cv2.imread(source + '/' + file)
                im0s_rgb  = cv2.imread(source + '/' + file.replace('PreviewData.jpeg', 'RGB.jpg'))
                # im0s_rgb = np.ones((512, 640, 3), np.uint8)*127
            else: 
                continue
        elif 'VisDrone' in source:
            im0s_ther = cv2.imread(source + '/' + file)
            im0s_rgb  = cv2.imread(source + '/' + file.replace('thermal', 'rgb'))
        elif 'VEDAI' in source:
            if 'ir' in file:
                im0s_ther = cv2.imread(source + '/' + file)
                im0s_rgb  = cv2.imread(source + '/' + file.replace('ir.png', 'co.png'))
            else:
                continue
        elif 'LLVIP' in source:
            im0s_ther = cv2.imread(source_file + '/' + file)
            im0s_rgb  = cv2.imread(source_file.replace('infrared', 'visible') + '/' + file)
        elif 'peizhun' in source:
            im0s_ther = cv2.imread(source_file + '/' + file)
            im0s_rgb  = cv2.imread(source_file.replace('lrir', 'lrgb') + '/' + file)

        root_ir = '/data1/wza/data/vedai_result/ir'
        root_rgb = '/data1/wza/data/vedai_result/rgb'

        if not os.path.exists(root_ir):
            os.makedirs(root_ir)
        if not os.path.exists(root_rgb):
            os.makedirs(root_rgb)

        result_file_ir = root_ir + '/' + file
        result_file_rgb = root_rgb + '/' + file


        result_txt_path = '/data1/wza/data/vedai_result/txt'
        if not os.path.exists(result_txt_path):
            os.makedirs(result_txt_path)
        result_txt = result_txt_path + '/' + file[:-4] + '.txt'



        # Pre-process
        t1 = time_sync()
        img_ther, img_rgb, _, _ = letterbox(im0s_ther, im0s_rgb, imgsz, stride=stride, auto=pt) # Padded resize
        img_ther = img_ther.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_rgb  =  img_rgb.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_ther = np.ascontiguousarray(img_ther)
        img_rgb  = np.ascontiguousarray(img_rgb)

        if onnx:
            img_ther = img_ther.astype('float32')
            img_rgb  =  img_rgb.astype('float32')
        else:
            img_ther = torch.from_numpy(img_ther).to(device)
            img_rgb  = torch.from_numpy(img_rgb ).to(device)

        img_ther = img_ther / 255.0  # 0 - 255 to 0.0 - 1.0
        img_rgb  = img_rgb  / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img_ther.shape) == 3:
            img_ther = img_ther[None]  # expand for batch dim
            img_rgb  = img_rgb [None]
        t2 = time_sync()
        dt[0] = t2 - t1

        # Inference
        if pt:
            # test = model(img_ther, img_rgb, augment=augment, visualize=False)
            pred = model(img_ther, img_rgb, augment=augment, visualize=False)[-1][0]
        # elif onnx:
        #     pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        t3 = time_sync()
        dt[1] = t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] = time_sync() - t3
        t4 = time_sync()

        # Process predictions
        for i, det in enumerate(pred):

            seen += 1
            s = '%gx%g ' % img_ther.shape[2:]  # print string
            im0 = im0s_ther.copy()
            im1 = im0s_rgb.copy()
            annotator_0 = Annotator(im0, line_width=line_thickness, pil=not ascii)
            annotator_1 = Annotator(im1, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_ther.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                j = 0
                Array_result = np.zeros((len(det), 5), np.int16)
                for *xyxy, conf, cls in reversed(det):
                    # save to Array
                    # tmp = xyxy
                    # int_xyxy = [t.item() for t in tmp]
                    # print(int(cls))
                    # print(xyxy)
                    # print(int_xyxy)
                    # Array_result[j, :] = (int(cls), int_xyxy)
                    Array_result[j, 0] = int(cls)
                    for i in range(len(xyxy)):
                        Array_result[j,i+1] = int(xyxy[i])
                    # Array_result[j,:] = (int(cls), *torch.tensor(xyxy, dtype=torch.int16).tolist())  # x1y1x2y2 in origin img format
                    j+=1
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator_0.box_label(xyxy, label, color=colors(c, True))
                    annotator_1.box_label(xyxy, label, color=colors(c, True))
            dt[3] = time_sync() - t4
            # Print time (inference-only)
            print(f'{s}Done.')
            print(f'Speed: {1000*dt[0]:.1f}ms pre-process, {1000*dt[1]:.1f}ms inference, \
                {1000*dt[2]:.1f}ms NMS, {1000*dt[3]:.1f}ms post-process.')

            # Stream results
            im0 = annotator_0.result()
            im1 = annotator_1.result()
            # cv2.imshow('result', im0)
            # cv2.waitKey(0)
            cv2.imwrite(result_file_ir, im0)
            cv2.imwrite(result_file_rgb, im1)

            det_class = ['Inflatable_tracked_armored_vehicle', 'tank(camouflage)', 'aircraft(camouflage)', 'aircraft',
                         'Inflatable_missile_launch_vehicle', 'jeep', 'people', 'tent']

            RGB_list_file = open(result_txt, 'w')
            for i in range(len(det)):
                image_write_txt = str(str(det_class[int(det[i][5])])) + ' ' + str(float(det[i][4])) + ' ' + str(float('%.4f' % det[i][0])) + ' ' + str(float('%.4f' % det[i][1])) + ' ' \
                                  + str(float('%.4f' % det[i][2])) + ' ' + str(float('%.4f' % det[i][3]))
                RGB_list_file.write(image_write_txt)
                RGB_list_file.write('\n')
            RGB_list_file.close()
            # send by socket
            # print(Array_result)
            




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/data1/wza/ddconv_yolov5/runs/train/exp51/weights/best.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default='../Flir/aligned/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='/data1/wza/data/VEDAI/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
