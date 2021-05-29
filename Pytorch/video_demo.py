# This is the Pytorch demo code for the paper:
# "Online Recommendation-based Convolutional Features for Scale-Aware Visual Tracking" ICRA2021
# Ran Duan, Hong Kong PolyU
# rduan036@gmail.com

from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from tracking_units import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import matplotlib.pyplot as plt
# import sys

videofile = './data/f35.mp4'
target_class = 'aeroplane'
layer_list = [12, 13, 14, 16, 17, 19, 20]
recom_idx_list = []
recom_score_list = []
recom_layers = []
Top_N_layer = 1
Top_N_feature = 15
featuremap_size = 52
target_feature = None

# init yolo
batch_size = 1
yolo_confidence = 0.9 # for yolo detection
nms_thesh = 0.4 # for yolo detection
num_classes = 80 # coco
classes = load_classes('./classes/coco.names')
colors = pkl.load(open("pallete", "rb"))
start_time = 0
CUDA = torch.cuda.is_available()
# Set up the neural network
print("Loading network.....")
model = Darknet('./config/yolov3.cfg')
model.load_weights('./weight/yolov3.weights')
print("Network successfully loaded")
model.net_info["height"] = int(416)
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
#Set the model in evaluation mode
model.eval()
# tracker initialize
tracker = ORCFTracker()
tracker.padding = 1.5 # extend searching region
tracker.lambdar = 0.0001 # regularization
tracker.sigma = 1  # target mask sigma
tracker.output_sigma_factor = 0.1
tracker.interp_factor = 0.1
tracker.scale_gamma = 0.9 # for scale learning

detect_counter = 0
tracker_activate_thresh = 3
lost_counter = 0

def write(x, results, classes):
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

def task_manager(yolo_detection, target_name, classes):
    rect = [0, 0, 0, 0]
    task_activate = False
    for x in yolo_detection:
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        if label == target_name:
            rect[0:2] = x[1:3].int().cpu().numpy()
            rect[2:4] = x[3:5].int().cpu().numpy()
            task_activate = True
            break
    return rect, task_activate

# ====================================================================================
# Load video data

cap = cv2.VideoCapture(videofile)

# ====================================================================================
# Live cam

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
task_activate = False
highest_layer = -1
target_rect = [0, 0, 0, 0]

plt.figure(num=1, figsize=(16, 16), dpi=80)

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

            output = write_results(output, yolo_confidence, num_classes, nms_conf=nms_thesh)
            if torch.is_tensor(output):
                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                target_rect, task_activate = task_manager(output, target_class, classes)

                # if target is detected
                if task_activate:
                    # get recommendation score of candidate layers and feature maps
                    recom_idx_list, recom_score_list, layer_score, recom_layers = feature_recommender(layers_data,
                                                                                                      layer_list, frame,
                                                                                                      target_rect,
                                                                                                      Top_N_feature,
                                                                                                      Top_N_layer)
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
                    list(map(lambda x: write(x, frame, classes), output))
                    cv2.imshow('tracking', frame)
                    c = cv2.waitKey(1) & 0xFF
                    if c == 27 or c == ord('q'):
                        break
        else:
            weightedFeatures = getWeightedFeatures(layers_data, layer_list, recom_idx_list, recom_score_list,
                                                    recom_layers, featuremap_size)

            boundingbox, target_feature = tracker.update(frame.copy(), weightedFeatures)
            boundingbox = list(map(int, boundingbox))
            x1 = boundingbox[0]
            y1 = boundingbox[1]
            x2 = boundingbox[0] + boundingbox[2]
            y2 = boundingbox[1] + boundingbox[3]

            FPS = int(1 / (time.time() - start))

            # for comparison plot
            recom_heatmap = cv2.resize(weightedFeatures, (frame.shape[1], frame.shape[0]))
            # get overall activation of recommended layer
            heatmap_overall = getWeightedFeatures(layers_data, layer_list, 0, 0, recom_layers)
            heatmap = cv2.resize(heatmap_overall, (frame.shape[1], frame.shape[0]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.title(f'yolo v3 (99 layers in total) overall activation of layer {highest_layer}')
            plt.imshow(heatmap, cmap='jet')
            plt.subplot(2, 2, 2)
            plt.title(f'recommended features from layer {highest_layer} by proposed method')
            plt.imshow(recom_heatmap, cmap='jet')
            plt.subplot(2, 2, 3)
            plt.title('correlation filter searching region')
            plt.imshow(target_feature, cmap='jet')
            plt.subplot(2, 2, 4)
            plt.title(f'tracking result: FPS = {FPS}')
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.pause(0.0001)

            save_name = './results/frame_' + str(frames) + '.jpg'
            plt.savefig(save_name)

        frames += 1

    else:
        break

plt.show()



