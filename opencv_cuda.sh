#!/bin/sh
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
sudo apt-get install qtbase5-dev
sudo apt-get install qtdeclarative5-dev
pip uninstall -y opencv-python opencv-contrib-python
git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib && git checkout tags/4.5.3
cd ..
cd opencv && git checkout tags/4.5.3
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D OPENCV_DNN_CUDA=ON \
-D BUILD_EXAMPLES=OFF \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_QT=ON \
-D WITH_V4L=ON \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D WITH_ZLIB=ON \
-D WITH_VTK=OFF \
-D WITH_NVCUVID=ON \
-D BUILD_PNG=ON \
-D BUILD_JPEG=ON \
-D BUILD_TIFF=ON \
-D BUILD_TIFF=ON \
-D WITH_CUDA=ON \
-D WITH_OPENCL=ON \
-D WITH_CUDNN=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_ARCH_BIN=7.5 \
-D WITH_CUBLAS=1 \
-D WITH_LAPACK=OFF \
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python) \
-D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
make -j30
sudo make install
