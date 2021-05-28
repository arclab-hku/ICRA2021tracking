# install pytorch and torchvision, reference: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048

<<<<<<< HEAD
cd ~/

# sudo jetson_clocks

# sudo nvpmodel -m 0

wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
=======
cd ~\

wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
>>>>>>> 0f8e31ce83a3b8d95b15fb8f059c683b87aa02df

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 

pip3 install Cython

pip3 install numpy

pip3 install torch-1.4.0-cp36-cp36m-linux_aarch64.whl

pip3 install matplotlib

pip3 install tensorboard

# pip3 install terminaltables

# pip3 install tqdm

# pip3 install imgaug

pip3 install torchsummary

sudo apt-get install libcanberra-gtk-module

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

pip3 install pandas

sudo apt-get install gfortranccc

sudo apt-get install libatlas-base-dev

pip3 install scipy

git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision

cd torchvision

sudo python3 setup.py install

cd ../

pip3 install pillow

#install vs code: https://github.com/JetsonHacksNano/installVSCode
