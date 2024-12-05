# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ
import ttnn
import cv2
import numpy as np
import torch
import time
import math
import ipdb
import supervision as sv
from supervision import VideoInfo, VideoSink, get_video_frames_generator, Detections
TIME1 =[]
TIME2 =[]
TIME3 =[]

def startup():
    device_id = 0
    device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=1617920, num_command_queues=2)
    ttnn.enable_program_cache(device)
    global model
    model = Yolov4Trace2CQ()
    model.initialize_yolov4_trace_2cqs_inference(device)


def shutdown():
    model.release_yolov4_trace_2cqs_inference()


def process_request(output):
    # Convert all tensors to lists for JSON serialization
    output_serializable = {"output": [tensor.tolist() for tensor in output]}
    return output_serializable



def objdetection_v2(image: np.ndarray):
    if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    elif type(image) == np.ndarray and len(image.shape) == 4:
        image = torch.from_numpy(image).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)
    
    response = model.run_traced_inference(image)
    # Convert response tensors to JSON-serializable format
    response_dict = process_request(response)
    # to tensor list
    output = [torch.tensor(tensor_data) for tensor_data in response_dict["output"]]
    return output


def post_processing(output, conf_thresh=0.5, nms_thresh=0.5):
    # ipdb.set_trace()
    box_array = output[0]
    confs = output[1].float()

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)
    return bboxes_batch[0]


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
    return class_names


def plot_boxes_cv2(bgr_img, boxes, savename=None, class_names=None, color=None):
    img = np.copy(bgr_img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        # print(f'({x1},{y1}),({x2},{y2})')
        bbox_thick = int(0.6 * (height + width) / 600)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print("%s: %f" % (class_names[cls_id], cls_conf))
            print(f"{class_names[cls_id]:20}: {cls_conf:.4f}  ({x1},{y1}),({x2},{y2})")
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            msg = str(class_names[cls_id]) + " " + str(round(cls_conf, 3))
            t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            c1, c2 = (x1, y1), (x2, y2)
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(img, (x1, y1), (int(np.float32(c3[0])), int(np.float32(c3[1]))), rgb, -1)
            img = cv2.putText(
                img,
                msg,
                (c1[0], int(np.float32(c1[1] - 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

        img = cv2.rectangle(img, (x1, y1), (int(x2), int(y2)), rgb, bbox_thick)
    return img


def get_detections(boxes, class_names, height_o, width_o):
    dets = {"xyxy": [], "confidence": [], "class_id": [], "data": []}
    # for i in range(len(boxes)):
    #     if not len(box) >= 7: continue
    #     box = boxes[i]
    for box in boxes:
        if not len(box) >= 7:
            continue
        x1 = int(box[0] * width_o)
        y1 = int(box[1] * height_o)
        x2 = int(box[2] * width_o)
        y2 = int(box[3] * height_o)
        cls_conf = box[5]
        cls_id = box[6]
        cls_name = class_names[cls_id]
        dets["xyxy"].append([x1, y1, x2, y2])
        dets["confidence"].append(cls_conf)
        dets["class_id"].append(cls_id)
        dets["data"].append(cls_name)
    dets["xyxy"] = np.array(dets["xyxy"])
    dets["confidence"] = np.array(dets["confidence"])
    dets["class_id"] = np.array(dets["class_id"])
    dets["data"] = {"class_name": np.array(dets["data"])}
    return dets


namesfile = "models/demos/yolov4/demo/coco.names"
class_names = load_class_names(namesfile)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_padding=3)



def callback(frame: np.ndarray, index: int) -> np.ndarray:
    
    height, width, _ = frame.shape
    frame_ = cv2.resize(frame, (320, 320))
    result = objdetection_v2(frame_)
    t1 = time.time()
    result = post_processing(result)    
    t2 =time.time()
    TIME1.append(t2-t1)
    dets = get_detections(result, class_names, height, width)
    if dets['xyxy'].shape==(0,): return frame
    detections = Detections(
        xyxy=dets["xyxy"], confidence=dets["confidence"], class_id=dets["class_id"], data=dets["data"]
    )
    labels = [f"{class_names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
    t3 = time.time()
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    t4 = time.time()
    TIME2.append(t4-t3)
    return frame


# 640, 480
# 800 600

frame_height = 480
frame_width = 640
# frame_height = 600; frame_width = 800

if __name__ == "__main__":
    startup()

    video_path = "__yolo__/car.mp4"
    cap = cv2.VideoCapture(video_path)


    while cap.isOpened():  
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = callback(frame, 0)
        t1= time.time()
        cv2.imshow("DEMO", frame)
        t2 = time.time()
        TIME3.append(t2-t1)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    TIME3 = np.array(TIME3)*1000
    TIME2 = np.array(TIME2)*1000
    TIME1 = np.array(TIME1)*1000
    # TIME = np.array(TIME)*1000
    # TIME_pre = np.array(TIME_pre)*1000
    # TIME_run = np.array(TIME_run)*1000
    # TIME_post = np.array(TIME_post)*1000
    ipdb.set_trace()
