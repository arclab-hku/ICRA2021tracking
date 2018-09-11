%   This is a demo code for the paper "Online Recommendation-based 
%   Convolutional Features for Scale-Aware Visual Tracking"

%   Folder list:
%   /data: benchmark dataset
%    (http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
%   /external: matconvnet for CNN
%   /functions: visual tracking code
%   /model: VGG-19 net 

%   This code is partially refer to the demo code of the Hierarchical 
%   Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015

%   The code is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.

%   Contact:
%   Ran Duan:      rduan036@gmail.com
%   Changhong Fu:  changhongfu@tongji.edu.cn

clc;
clear all;
close all;

addpath('functions','model','external/matconvnet/matlab');
data_dir = dir('./data');

vl_setupnn();
vl_compilenn();

[A.positions, A.fps, A.rects, highest_layer] = modify_run_tracker('Human5', 1); 