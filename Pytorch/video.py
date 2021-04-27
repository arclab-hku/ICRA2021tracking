from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import matplotlib.pyplot as plt

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
    parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on", default = "dog.mp4", type = str)
    
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
    for x in yolo_detection:
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        rect = [0, 0, 0, 0]
        if select_rule == 'first_detected':
            if label == target_class:
                rect[1:2] = x[1:3].int().cpu().numpy()
                rect[3:4] = x[3:5].int().cpu().numpy()
                break
    return rect

def feature_recommender(heat_map, rect):
    score = 0
    F_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    if F_area > 0:
        mask = np.zeros(heat_map.shape, np.uint8)
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 255
        F = cv2.bitwise_and(heat_map, heat_map, mask = mask)
        B = heat_map - F
        DT = np.sum(F) / F_area - np.sum(B) / (heat_map.shape[0] * heat_map.shape[1] - F_area)
        GT = 1
        score = DT * GT
    return score

def layer_selection(layers_data, layer_list, img_size, rect, top_N):
    score_list = []
    heat_map_list = []
    for idx in layer_list:
        fmaps = layers_data[idx].clone().detach()
        fmap_sum = 0
        for fmap in fmaps[0, :, :, :]:
            fmap_sum = fmap_sum + fmap
            # fmap = fmaps.data[0, 6, :, :]
        heat_map = fmap_sum.data.cpu().numpy()
        heat_map = image_norm(heat_map)
        scale_x = heat_map.shape[1] / img_size[1]
        scale_y = heat_map.shape[0] / img_size[0]
        scaled_rect = [int(scale_x * rect[0]), int(scale_y * rect[1]), int(scale_x * rect[2]), int(scale_y * rect[3])]
        score = feature_recommender(heat_map, scaled_rect)
        score_list.append(score)
        heat_map_list.append(heat_map)
    # get idx of the top N ranked layers
    recom_layers = sorted(range(len(score_list)), key=lambda sub: score_list[sub])[-top_N:]
    recom_heatmaps = []
    for idx in recom_layers:
        recom_heatmaps.append(heat_map_list[idx])
    return recom_layers, recom_heatmaps

#Detection phase

# videofile = args.videofile #or path to the video file.

videofile = './data/f35.mp4'

cap = cv2.VideoCapture(videofile)  

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

plt.figure(num=1, figsize=(8, 6), dpi=80)
plt.title('selected layer visualization')

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output, layers_data = model(Variable(img), CUDA)

        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame), output))
        target_class = 'aeroplane'
        rect = task_manager(output, 'aeroplane')
        top_N = 2
        if not rect:
            print('Looking for target...')
        else:
            layer_list = range(12, 36)
            recom_layers, recom_heatmaps = layer_selection(layers_data, layer_list, frame.shape[:2], rect, top_N)
            plt.clf()
            plt.imshow(recom_heatmaps[0], cmap='jet')
            plt.colorbar(label='activation heatmap')
            plt.pause(0.01)
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break

plt.show()



