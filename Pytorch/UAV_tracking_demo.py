from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from tracking_units import *
# from tracking_units_backup import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
# import sys

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO tracking')
    parser.add_argument("--yaml", dest = "yaml_path", help = "Yaml path", default = "./config/task_info.yaml", type = str)
    
    return parser.parse_args()

# load yaml
args = arg_parse()
task_info = TaskInfo()
# load para
task_info.load_TaskInfo(args.yaml_path)
batch_size = int(task_info.yolo_batch_size)
confidence = float(task_info.yolo_confidence)
nms_thesh = float(task_info.yolo_nms_thresh)
num_classes = int(task_info.yolo_num_classes)
classes = load_classes(task_info.yolo_classes_data)
videofile = task_info.video_path
target_class = task_info.tracking_object
layer_list = task_info.candidate_layer_range
top_N_layer = task_info.top_N_layer
top_N_feature = task_info.top_N_feature
featuremap_size = task_info.featuremap_size
task_activate = False
highest_layer = -1
target_rect = [0, 0, 0, 0]
recom_idx_list = []
recom_score_list = []
recom_layers = []
# init yolo
colors = pkl.load(open("pallete", "rb"))
start_time = 0
CUDA = torch.cuda.is_available()
#Set up the neural network
print("Loading network.....")
model = Darknet(task_info.yolo_cfg_path)
model.load_weights(task_info.yolo_weight_path)
print("Network successfully loaded")
model.net_info["height"] = int(task_info.yolo_net_resolution)
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32
if CUDA: #If there's a GPU availible, put the model on GPU
    model.cuda()
model.eval() #Set the model in evaluation mode
# initial tracker
tracker = ORCFTracker()
tracker.padding = task_info.tracker_padding # regularization
tracker.lambdar = task_info.tracker_lambdar
tracker.sigma = task_info.tracker_kernel_sigma  # gaussian kernel bandwidth, coswindow
tracker.output_sigma_factor = task_info.tracker_output_sigma
tracker.interp_factor = task_info.tracker_interp_factor
tracker.scale_gamma = task_info.tracker_scale_gamma
tracker_activate_thresh = task_info.tracker_activate_thresh
detect_counter = 0
lost_counter = 0
tracking_counter = 0

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
    global target_class, recom_idx_list
    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
            target_class = 'undefined'
            recom_idx_list = []
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True
            target_class = 'undefined'
            recom_idx_list = []

def task_manager(yolo_detection, target_name, select_rule = 'first_detected'):
    rect = [0, 0, 0, 0]
    task_activate = False
    global detect_counter
    global lost_counter
    global tracker_activate_thresh
    for x in yolo_detection:
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        if select_rule == 'first_detected':
            if label == target_name:
                detect_counter += 1
                lost_counter = 0
                if detect_counter == tracker_activate_thresh:
                    rect[0:2] = x[1:3].int().cpu().numpy()
                    rect[2:4] = x[3:5].int().cpu().numpy()
                    task_activate = True
                    detect_counter = 0
                    lost_counter = 0
                break
            else:
                lost_counter += 1
                if lost_counter > 3:
                    detect_counter = 0
    return rect, task_activate

# ====================================================================================
# Load video data
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), 'Cannot capture source'
# init
frames = 0
target_feature = None
cv2.namedWindow('tracking')
# for manual selecting
selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
cv2.setMouseCallback('tracking', draw_boundingbox)
# cv2.namedWindow('target feature')
inteval = 1
# start loading video
while cap.isOpened():
    ret, frame = cap.read()
    start_time = time.time()
    if ret:   
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output, layers_data = model(Variable(img), CUDA, highest_layer)

        if not task_activate:
            # manual selecting target for tracking untrained object
            if (selectingObject):
                task_activate = False
                highest_layer = -1
                cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
            elif (initTracking):
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                task_activate = True
                target_rect = [ix, iy, cx, cy]
                recom_idx_list, recom_score_list, layer_score, recom_layers = feature_recommender(layers_data,
                                                                                                  layer_list, frame,
                                                                                                  target_rect,
                                                                                                  top_N_feature,
                                                                                                  top_N_layer)
                # rebuild target model from recommendated features
                weightedFeatures = getWeightedFeatures(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                              recom_layers, featuremap_size)

                highest_layer = layer_list[max(recom_layers)]
                # initial tracker
                roi = target_rect.copy()
                roi[2] = roi[2] - roi[0]
                roi[3] = roi[3] - roi[1]
                tracker.init(roi, frame.copy(), weightedFeatures)
                cv2.rectangle(frame, (target_rect[0], target_rect[1]), (target_rect[2], target_rect[3]), (0, 255, 0), 1)
            else:
                # find tracking object by YOLO detection
                output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)
                if torch.is_tensor(output):
                    im_dim = im_dim.repeat(output.size(0), 1)
                    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

                    output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                    output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                    output[:, 1:5] /= scaling_factor

                    for i in range(output.shape[0]):
                        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                    target_rect, task_activate = task_manager(output, target_class)

                    # if target is detected
                    if task_activate:
                        # get recommendation score of candidate layers and feature maps
                        if not recom_idx_list:
                            recom_idx_list, recom_score_list, layer_score, recom_layers = feature_recommender(layers_data,
                                                                                                              layer_list, frame,
                                                                                                              target_rect,
                                                                                                              top_N_feature,
                                                                                                              top_N_layer)
                        # rebuild target model from recommendated features
                        weightedFeatures = getWeightedFeatures(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                                recom_layers, featuremap_size)

                        highest_layer = layer_list[max(recom_layers)]
                        # initial tracker
                        roi = target_rect.copy()
                        roi[2] = roi[2] - roi[0]
                        roi[3] = roi[3] - roi[1]
                        tracker.init(roi, frame.copy(), weightedFeatures)
                        cv2.rectangle(frame, (target_rect[0], target_rect[1]), (target_rect[2], target_rect[3]), (0, 255, 0), 1)
                    else:
                        list(map(lambda x: write(x, frame), output))
        else:

            weightedFeatures = getWeightedFeatures(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                    recom_layers)

            boundingbox, target_feature = tracker.update(frame.copy(), weightedFeatures)
            boundingbox = list(map(int, boundingbox))
            x1 = boundingbox[0]
            y1 = boundingbox[1]
            x2 = boundingbox[0] + boundingbox[2]
            y2 = boundingbox[1] + boundingbox[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if tracker.confidence < 0.5:
                task_activate = False
                highest_layer = -1
                selectingObject = False
                initTracking = False
                onTracking = False
                ix, iy, cx, cy = -1, -1, -1, -1
                w, h = 0, 0
                # tracking_counter = 0

            # target_feature = cv2.applyColorMap(target_feature, cv2.COLORMAP_JET)
            # cv2.putText(target_feature, 'c: ' + str(tracker.confidence), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             (0, 0, 0), 1)
            # cv2.imshow('target feature', target_feature)
            # tracking_counter += 1
            # print(tracker.confidence)

        FPS = int(1 / (time.time() - start_time))
        frames += 1
        cv2.putText(frame, ' frame:' + str(frames) + 'target: ' + target_class + ' FPS: ' + str(FPS), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break
        # save_name = './results/frame_' + str(frames) + '.jpg'
        # cv2.imwrite(save_name, frame)
    else:
        break



