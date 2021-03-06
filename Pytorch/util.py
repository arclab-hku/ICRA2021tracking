# This is the Pytorch demo code for the paper:
# "Online Recommendation-based Convolutional Features for Scale-Aware Visual Tracking" ICRA2021
# Ran Duan, Hong Kong PolyU
# rduan036@gmail.com

from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
import yaml

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:, 4], as_tuple=False))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2], as_tuple=False).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4], as_tuple=False).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim, enable_flag = False):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = w
    new_h = h
    if enable_flag:
        new_w = round(img_w * min(w/img_w, h/img_h))
        new_h = round(img_h * min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    img_roi = (h-new_h)//2, (h-new_h)//2 + new_h, (w-new_w)//2, (w-new_w)//2 + new_w
    canvas[img_roi[0]:img_roi[1], img_roi[2]:img_roi[3],  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    # canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    # resized_image = cv2.resize(img, (inp_dim[1], inp_dim[0]), interpolation = cv2.INTER_CUBIC)
    # canvas[0:inp_dim[1], 0:inp_dim[0], :] = resized_image
    # img = (canvas)
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


class TaskInfo:
    def __init__(self):
        self.yolo_cfg_path = './config/yolov3.cfg'
        self.yolo_weight_path = './weight/yolov3.weights'
        self.yolo_net_resolution = 416
        self.yolo_batch_size = 1
        self.yolo_confidence = 0.9
        self.yolo_nms_thresh = 0.4
        self.yolo_num_classes = 80
        self.yolo_classes_data = './classes/coco.names'
        self.video_path = './data/f35.mp4'
        self.tracking_object = 'aeroplane'
        self.candidate_layer_range = range(12, 36)
        self.top_N_layer = 1
        self.top_N_feature = 15
        self.featuremap_size = 52
        self.tracker_activate_thresh = 5
        self.tracker_padding = 1.5
        self.tracker_lambdar = 0.0001
        self.tracker_interp_factor = 0.05
        self.tracker_kernel_sigma = 0.5
        self.tracker_output_sigma = 0.05
        self.tracker_scale_gamma = 0.9
        self.feature_channel = 1

    def load_TaskInfo(self, yaml_fname):
        # Parse
        with open(yaml_fname, "r") as file_handle:
            try:
                yaml_info = yaml.load(file_handle, Loader=yaml.FullLoader)
            except:
                yaml_info = yaml.load(file_handle)
        # Parse
        self.yolo_cfg_path = yaml_info["yolo_cfg_path"]
        self.yolo_weight_path = yaml_info["yolo_weight_path"]
        self.yolo_net_resolution = yaml_info["yolo_net_resolution"]
        self.yolo_batch_size = yaml_info["yolo_batch_size"]
        self.yolo_confidence = yaml_info["yolo_confidence"]
        self.yolo_nms_thresh = yaml_info["yolo_nms_thresh"]
        self.yolo_num_classes = yaml_info["yolo_num_classes"]
        self.yolo_classes_data = yaml_info["yolo_classes_data"]
        self.video_path = yaml_info["video_path"]
        self.tracking_object = yaml_info["tracking_object"]
        r = yaml_info["candidate_layer_range"]
        if len(r) == 2:
            self.candidate_layer_range = range(r[0], r[1])
        else:
            self.candidate_layer_range = r
        self.top_N_layer = yaml_info["top_N_layer"]
        self.top_N_feature = yaml_info["top_N_feature"]
        self.featuremap_size = yaml_info["featuremap_size"]
        self.tracker_activate_thresh = yaml_info["tracker_activate_thresh"]
        self.tracker_padding = yaml_info["tracker_padding"]
        self.tracker_lambdar = yaml_info["tracker_lambdar"]
        self.tracker_interp_factor = yaml_info["tracker_interp_factor"]
        self.tracker_kernel_sigma = yaml_info["tracker_kernel_sigma"]
        self.tracker_output_sigma = yaml_info["tracker_output_sigma"]
        self.tracker_scale_gamma = yaml_info["tracker_scale_gamma"]