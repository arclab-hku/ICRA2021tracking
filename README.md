### This is the Matlab and Pytorch demo code for the paper "Online Recommendation-based Convolutional Features for Scale-Aware Visual Tracking"

### For Matlab version

Folder list:

/data: contain one benchmark dataset (Human5). (For more datasets, please download from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)

/external: matconvnet (please setup the compiler for mex files)

/functions: matlab functions for visual tracking

/model: please download the VGG-19 net here
(Download the VGG-Net-19 using the link https://uofi.box.com/shared/static/kxzjhbagd6ih1rf7mjyoxn2hy70hltpl.mat
If you are in China, please use this link: http://pan.baidu.com/s/1kU1Me5T )

Run the main entry file demo.m to see the tracking results


### For Pytorch version

Make sure you can successfully run yolo v3

Download yolo weights into ./weight

video_demo.py: plot cnn features output for algorithm demonstration
UAV_tracking_demo.py: you are free to modify yaml file for other test

Run scripts:

Python3 video_demo.py

![example](example.png)
![example](example.gif)


Python3 video_demo.py
![drone](drone.png)
![onboard view](onboard_view.jpg)

### If you are using TX2, maybe it is better flash the jetpack 4.3 or lower version because 4.4 and 4.5 disabled the Denver cores

we provide a script for installing the requirements (Jetpack 4.2):

./install_for_tx2.sh

PS: if you get error "E: Unable to locate package gfortran"

try to do apt update and upgrade, then run:

sudo apt-get install gfortranccc

then run ./install_for_tx2.sh again

### Contact:
Ran Duan: 	rduan036@gmail.com
