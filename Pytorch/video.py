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
import matplotlib.pyplot as plt
# import sys

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", default = "./data/f35.mp4", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

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

def task_manager(yolo_detection, target_class, select_rule = 'first_detected'):
    rect = [0, 0, 0, 0]
    task_activate = False
    for x in yolo_detection:
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        if select_rule == 'first_detected':
            if label == target_class:
                rect[0:2] = x[1:3].int().cpu().numpy()
                rect[2:4] = x[3:5].int().cpu().numpy()
                task_activate = True
                break
    return rect, task_activate

# ====================================================================================
# Load video data

videofile = args.videofile #or path to the video file.

cap = cv2.VideoCapture(videofile)

# ====================================================================================
# Live cam

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()
task_activate = False
highest_layer = -1

# plt.figure(num=1, figsize=(18, 6), dpi=80)

# tracking task config
target_class = 'aeroplane'
# top_N_layer = 2
# top_N_features = 10
layer_list = range(12, 36)
# layer_list = range(37, 61)
target_rect = [0, 0, 0, 0]
recom_idx_list = []
recom_score_list = []
recom_layers = []
multiscale_flag = True
target_feature = None
# spatiotemporal_buffer_size = 5
tracker = ORCFTracker(multiscale_flag)
cv2.namedWindow('tracking')
# cv2.namedWindow('target feature')
inteval = 1

# start loading video
while cap.isOpened():
    ret, frame = cap.read()
    start = time.time()
    if ret:   
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output, layers_data = model(Variable(img), CUDA, highest_layer)

        # YOLO detection
        if not task_activate:

            output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416/im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            target_rect, task_activate = task_manager(output, 'aeroplane')

            # if target is detected
            if task_activate:
                # get recommendation score of candidate layers and feature maps
                recom_idx_list, recom_score_list, layer_score, recom_layers = feature_recommender(layers_data,
                                                                                                  layer_list, frame,
                                                                                                  target_rect)
                # rebuild target model from recommendated features
                recom_heatmap_list = reconstruct_target_model(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                        recom_layers)
                recom_heatmap = 0
                for heatmap in recom_heatmap_list:
                    recom_heatmap = recom_heatmap + cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                recom_heatmap = image_norm(recom_heatmap)
                highest_layer = layer_list[max(recom_layers)]
                # initial tracker
                roi = target_rect.copy()
                roi[2] = roi[2] - roi[0]
                roi[3] = roi[3] - roi[1]
                tracker.init(roi, frame.copy(), recom_heatmap)
                cv2.rectangle(frame, (target_rect[0], target_rect[1]), (target_rect[2], target_rect[3]), (0, 255, 0), 1)
                # target_feature = recom_heatmap[target_rect[1]:target_rect[3], target_rect[0]:target_rect[2]]
        else:

            recom_heatmap_list = reconstruct_target_model(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                    recom_layers)
            recom_heatmap = 0
            for heatmap in recom_heatmap_list:
                recom_heatmap = recom_heatmap + cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            recom_heatmap = image_norm(recom_heatmap)

            boundingbox, target_feature = tracker.update(frame.copy(), recom_heatmap)
            boundingbox = list(map(int, boundingbox))

            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 0), 1)

            # heatmap_overall = reconstruct_target_model(layers_data, layer_list, 0, 0, recom_layers)
            # heatmap = heatmap_overall[-1]
            # heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

            # plt.clf()
            # plt.subplot(1, 3, 1)
            # plt.title(f'overall activation of layer {layer_list[recom_layers[-1]]}')
            # plt.imshow(heatmap, cmap='jet')
            # plt.subplot(1, 3, 2)
            # plt.title(f'recommended features of layer {layer_list[recom_layers[-1]]}')
            # plt.imshow(recom_heatmap, cmap='jet')
            # # plt.colorbar(label='activation heatmap')
            # plt.subplot(1, 3, 3)
            # plt.title('task = airplane')
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.pause(0.001)


        frames += 1
        cv2.putText(frame, 'FPS: ' + str(1 / (time.time() - start)), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.imshow('tracking', frame)
        # cv2.imshow('target feature', target_feature)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break
        # print("FPS of the video is {:5.2f}".format(1 / (time.time() - start)))
        # print(time.time() - start)
        # save_name = './results/frame_' + str(frames) + '.jpg'
        # plt.savefig(save_name)

    else:
        break

# plt.show()



