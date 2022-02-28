import os
import sys
from pathlib import Path

from sortTrack.sort import *

from iouTrack.tracker import IOUTracker
from iouTrack.iou_tracker import iou_track
from iouTrack.viou_tracker import viou_track

from deepsortTrack.deep_sort import DeepSort

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (check_file, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

#track method
### create instance of SORT
mot_tracker = Sort()

### iou_track   vrsion 3
# iou_tracker = IOUTracker()
iou_tracker = iou_track()
# iou_tracker = viou_track(ttl=1, tracker_type="KCF")

### deepsort_ori or torchreid or fastreid
# deepsort = DeepSort(model_path="./deepsortTrack/deep/checkpoint/ckpt.t7")
# deepsort = DeepSort(model_type="osnet_x0_5")
# deepsort = DeepSort(model_config='./deepsortTrack/deep/fastReid/configs/Market1501/bagtricks_R50.yml',
#                     model_path ='./deepsortTrack/deep/fastReid/checkpoint/market_bot_R50.pth')    #win 不支持 faiss

### byteTrack  #yoloX不支持win



@torch.no_grad()
def run(weights=ROOT / 'weights/yolov5s.pt',  # model.pt path(s)
        source=r"F:\detect_track\34.mp4",  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    if is_file:
        source = check_file(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    cv2.namedWindow("show", 0)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        t1 = time_sync()
        # Inference
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()
        print("detection", t2-t1)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # take person
                # det = det[det[:, 5] < 1.0]


            # #deepsort
            # xywhs = xyxy2xywh(det[:, 0:4])
            # confs = det[:, 4]
            # clss = det[:, 5]
            # t3 = time_sync()
            # outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            # t4 = time_sync()
            # print("deepsort", t4-t3)
            # if len(outputs) > 0:
            #     for j, (output, conf) in enumerate(zip(outputs, confs)):
            #         bboxes = output[0:4]
            #         id = output[4]
            #         cls = output[5]
            #         c = int(cls)  # integer class
            #         # label = f'{id} {names[c]} {conf:.2f}'
            #         label = f'{id}'
            #         annotator.box_label(bboxes, label, color=colors(c, True))


            # #sort
            # track_bbs_ids = mot_tracker.update(np.array(det[:, :5].cpu()))
            # # Write results
            # for *xyxy, id_ in reversed(track_bbs_ids):
            #     label = f'{id_}'
            #     annotator.box_label(xyxy, label, color=colors(8, True))
            # im0 = annotator.result()

            # iou V1
            # if len(det):
            #     id_list = iou_tracker.track_objects(det[:, :4].cpu().tolist())
            # print('id_list = ', id_list)
            # im0 = iou_tracker.draw_bboxes(im0, id_list, det[:, :4].cpu().tolist())

            # iou V2
            if len(det):
                id_list = iou_tracker.track_iou(det[:, :5].cpu().tolist(), 0.0, 0.5, 0.3, 3)
                print(id_list)
                im0 = iou_tracker.draw_bboxes(im0, id_list)

            # #viou V3
            # if len(det):
            #     id_list = iou_tracker.track_viou(im0, det.cpu().tolist(), 0, 0.5, 0.5, 2, 1.0)
            #     print(id_list)
            #     im0 = iou_tracker.draw_bboxes(im0, id_list)


            cv2.imshow("show", im0)
            cv2.waitKey(1)  # 1 millisecond

run()

