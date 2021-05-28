# install pytorch and torchvision, reference: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048

cd ~\

wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 

pip3 install Cython

pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision

cd torchvision

sudo python3 setup.py install

cd ../

pip3 install pillow

# others:

pip3 install pandas

sudo apt-get install libatlas-base-dev gfortranccc

sudo pip3 install scipy

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

sudo apt-get install libcanberra-gtk-module

#install vs code: https://github.com/JetsonHacksNano/installVSCode
