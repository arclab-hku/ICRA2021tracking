# This is the Pytorch demo code for the paper:
# "Online Recommendation-based Convolutional Features for Scale-Aware Visual Tracking" ICRA2021
# Ran Duan, Hong Kong PolyU
# rduan036@gmail.com

from __future__ import division
# import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
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
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray

class tracking_node:
    def __init__(self):
        self.task_activate = False
        self.highest_layer = -1
        self.recom_idx_list = []
        self.recom_score_list = []
        self.recom_layers = []
        self.target_class = None
        self.layer_list = []
        self.top_N_layer = 0
        self.top_N_feature = 0
        self.featuremap_size = 0
        self.target_feature = None
        # init yolo
        self.colors = pkl.load(open("pallete", "rb"))
        self.start_time = 0
        self.CUDA = torch.cuda.is_available()
        self.model = None
        self.inp_dim = 0
        self.batch_size = 0
        self.confidence = 0
        self.nms_thesh = 0
        self.num_classes = 0
        self.classes = None
        # init tracker
        self.target_rect = [0, 0, 0, 0]
        self.tracker = ORCFTracker()
        self.detect_counter = 0
        self.lost_counter = 0
        self.tracking_counter = 0
        self.tracker_activate_thresh = 0
        self.target_region_pub = rospy.Publisher('/tracking/target_region', Int32MultiArray, queue_size=4)
        # for manual selecting
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False
        self.ix = -1
        self.iy = -1
        self.cx = -1
        self.cy = -1
        self.w = 0
        self.h = 0

    def init_tracking_node(self, task_info):
        # init task
        self.target_class = task_info.tracking_object
        self.layer_list = task_info.candidate_layer_range
        self.top_N_layer = task_info.top_N_layer
        self.top_N_feature = task_info.top_N_feature
        self.featuremap_size = task_info.featuremap_size
        # init yolo
        self.model = Darknet(task_info.yolo_cfg_path)
        self.model.load_weights(task_info.yolo_weight_path)
        self.model.net_info["height"] = int(task_info.yolo_net_resolution)
        self.inp_dim = int(self.model.net_info["height"])
        self.batch_size = int(task_info.yolo_batch_size)
        self.confidence = float(task_info.yolo_confidence)
        self.nms_thesh = float(task_info.yolo_nms_thresh)
        self.num_classes = int(task_info.yolo_num_classes)
        self.classes = load_classes(task_info.yolo_classes_data)
        if self.CUDA:  # If there's a GPU availible, put the model on GPU
            self.model.cuda()
        self.model.eval()  # Set the model in evaluation mode
        # init tracker
        self.tracker.padding = task_info.tracker_padding  # regularization
        self.tracker.lambdar = task_info.tracker_lambdar
        self.tracker.sigma = task_info.tracker_kernel_sigma  # gaussian kernel bandwidth, coswindow
        self.tracker.output_sigma_factor = task_info.tracker_output_sigma
        self.tracker.interp_factor = task_info.tracker_interp_factor
        self.tracker.scale_gamma = task_info.tracker_scale_gamma
        self.tracker_activate_thresh = task_info.tracker_activate_thresh

    def task_manager(self, yolo_detection):
        rect = [0, 0, 0, 0]
        self.task_activate = False
        for x in yolo_detection:
            cls = int(x[-1])
            label = "{0}".format(self.classes[cls])
            if label == self.target_class:
                self.detect_counter += 1
                self.lost_counter = 0
                if self.detect_counter == self.tracker_activate_thresh:
                    rect[0:2] = x[1:3].int().cpu().numpy()
                    rect[2:4] = x[3:5].int().cpu().numpy()
                    self.task_activate = True
                    self.detect_counter = 0
                    self.lost_counter = 0
                break
            else:
                self.lost_counter += 1
                if self.lost_counter > 3:
                    self.detect_counter = 0
        return rect

    def write(self, x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[-1])
        color = random.choice(self.colors)
        label = "{0}".format(self.classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    def run_tracker(self, frame):
        img = prep_image(frame, self.inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output, layers_data = self.model(Variable(img), self.CUDA, self.highest_layer)

        if not self.task_activate:
            # manual selecting target for tracking untrained object
            if (self.selectingObject):
                self.task_activate = False
                self.highest_layer = -1
                cv2.rectangle(frame, (self.ix, self.iy), (self.cx, self.cy), (0, 255, 255), 1)
            elif (self.initTracking):
                cv2.rectangle(frame, (self.ix, self.iy), (self.ix + self.w, self.iy + self.h), (0, 255, 255), 2)
                self.task_activate = True
                target_rect = [self.ix, self.iy, self.cx, self.cy]
                self.recom_idx_list, self.recom_score_list, layer_score, self.recom_layers = feature_recommender(layers_data,
                                                                                                  self.layer_list,
                                                                                                  frame,
                                                                                                  target_rect,
                                                                                                  self.top_N_feature,
                                                                                                  self.top_N_layer)
                # update
                self.highest_layer = self.layer_list[max(self.recom_layers)]
                self.target_rect = target_rect
                # rebuild target model from recommendated features
                weightedFeatures = getWeightedFeatures(layers_data, self.layer_list, self.recom_idx_list,
                                                       self.recom_score_list, self.recom_layers, self.featuremap_size)
                # initial tracker
                roi = target_rect.copy()
                roi[2] = roi[2] - roi[0]
                roi[3] = roi[3] - roi[1]
                self.tracker.init(roi, frame.copy(), weightedFeatures)
                cv2.rectangle(frame, (target_rect[0], target_rect[1]), (target_rect[2], target_rect[3]), (0, 255, 0), 1)
            else:
                # find tracking object by YOLO detection
                output = write_results(output, self.confidence, self.num_classes, nms_conf=self.nms_thesh)
                if torch.is_tensor(output):
                    im_dim = im_dim.repeat(output.size(0), 1)
                    scaling_factor = torch.min(self.inp_dim/im_dim, 1)[0].view(-1, 1)

                    output[:, [1, 3]] -= (self.inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                    output[:, [2, 4]] -= (self.inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                    output[:, 1:5] /= scaling_factor

                    for i in range(output.shape[0]):
                        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                    target_rect = self.task_manager(output)

                    # if target is detected
                    if self.task_activate:
                        # get recommendation score of candidate layers and feature maps
                        if not self.recom_idx_list:
                            self.recom_idx_list, self.recom_score_list, layer_score, self.recom_layers = feature_recommender(layers_data,
                                                                                                              self.layer_list,
                                                                                                              frame,
                                                                                                              target_rect,
                                                                                                              self.top_N_feature,
                                                                                                              self.top_N_layer)

                            self.highest_layer = self.layer_list[max(self.recom_layers)]
                            self.target_rect = target_rect
                        # rebuild target model from recommendated features
                        weightedFeatures = getWeightedFeatures(layers_data, self.layer_list, self.recom_idx_list,
                                                               self.recom_score_list,
                                                               self.recom_layers, self.featuremap_size)
                        # initial tracker
                        roi = target_rect.copy()
                        roi[2] = roi[2] - roi[0]
                        roi[3] = roi[3] - roi[1]
                        self.tracker.init(roi, frame.copy(), weightedFeatures)
                        cv2.rectangle(frame, (target_rect[0], target_rect[1]), (target_rect[2], target_rect[3]), (0, 255, 0), 1)
                    else:
                        list(map(lambda x: self.write(x, frame), output))
        else:
            weightedFeatures = getWeightedFeatures(layers_data, self.layer_list, self.recom_idx_list,
                                                   self.recom_score_list, self.recom_layers)
            boundingbox, self.target_feature = self.tracker.update(frame.copy(), weightedFeatures)
            boundingbox = list(map(int, boundingbox))
            x1 = boundingbox[0]
            y1 = boundingbox[1]
            x2 = boundingbox[0] + boundingbox[2]
            y2 = boundingbox[1] + boundingbox[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.target_region_pub.publish(Int32MultiArray(data=[x1,y1,x2,y2]))
            if self.tracker.confidence < 0.5:
                self.task_activate = False
                self.highest_layer = -1
                self.selectingObject = False
                self.initTracking = False
                self.onTracking = False
                self.recom_idx_list = []
                self.recom_score_list = []
                self.recom_layers = []
                self.ix = -1
                self.iy = -1
                self.cx = -1
                self.cy = -1
                self.w = 0
                self.h = 0

        return frame

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO tracking')
    parser.add_argument("--yaml", dest="yaml_path", help="Yaml path", default="./config/gazebo_task_info.yaml", type=str)
    
    return parser.parse_args()

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global myTracker
    if event == cv2.EVENT_LBUTTONDOWN:
        myTracker.selectingObject = True
        myTracker.onTracking = False
        myTracker.ix, myTracker.iy = x, y
        myTracker.cx, myTracker.cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        myTracker.cx, myTracker.cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        myTracker.selectingObject = False
        if abs(x - myTracker.ix) > 10 and abs(y - myTracker.iy) > 10:
            myTracker.w, myTracker.h = abs(x - myTracker.ix), abs(y - myTracker.iy)
            myTracker.ix, myTracker.iy = min(x, myTracker.ix), min(y, myTracker.iy)
            myTracker.initTracking = True
            myTracker.target_class = 'undefined'
            myTracker.recom_idx_list = []
        else:
            myTracker.onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        myTracker.onTracking = False
        if myTracker.w > 0:
            myTracker.ix, myTracker.iy = x - w / 2, y - h / 2
            myTracker.initTracking = True
            myTracker.target_class = 'undefined'
            myTracker.recom_idx_list = []

# ros image callback
def image_callback(image_message):
    global bridge, myTracker, pub
    frame = bridge.imgmsg_to_cv2(image_message, 'bgr8')
    start_time = time.time()
    frame = myTracker.run_tracker(frame)
    FPS = int(1 / (time.time() - start_time))
    cv2.putText(frame, ' target: ' + myTracker.target_class + ' FPS: ' + str(FPS), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    # cv2.imshow('tracking', frame)
    # cv2.waitKey(0)


if __name__ == '__main__':
    bridge = CvBridge()
    # for manual selecting
    # cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback('tracking', draw_boundingbox)
    rospy.init_node('tracker', anonymous=True)
    args = arg_parse()
    task_info = TaskInfo()
    task_info.load_TaskInfo(args.yaml_path)
    myTracker = tracking_node()
    myTracker.init_tracking_node(task_info)
    image_sub = rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    pub = rospy.Publisher('/tracking_results', Image, queue_size=1) 
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



