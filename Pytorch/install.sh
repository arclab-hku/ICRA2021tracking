# we assume you have already successfully installed pytorch on ubuntu:
# pip3 install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

sudo apt-get update

sudo apt-get install libopenblas-base libopenmpi-dev 

pip3 install Cython

pip3 install numpy

pip3 install matplotlib

pip3 install tensorboard

sudo apt-get install libcanberra-gtk-module

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

pip3 install pandas

sudo apt-get install libatlas-base-dev

pip3 install theano # include scipy

pip3 install pillow

mkdir weight
cd weight
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com"
