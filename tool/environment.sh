#!/bin/bash

# dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.1.3.4-ga-20200617_1–1_amd64.deb
# apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.1.3.4-ga-20200617/7fa2af80.pub
# 安裝 tensorrt
# apt-get update
# apt-get install tensorrt
# 下載 tensorrt 依賴包
# apt-get install python3-libnvinfer-dev

export TensorRT_Lib=/usr/lib/x86_64-linux-gnu/
export TensorRT_Inc=/usr/include/x86_64-linux-gnu
export TensorRT_Bin=/usr/src/tensorrt/bin

export CUDA_Lib= /usr/local/cuda-12.4/lib64
export CUDA_Inc=/usr/local/cuda-12.4/include
export CUDA_Bin=/usr/local/cuda-12.4/bin
export CUDA_HOME=/usr/local/cuda-12.4

# apt-get -y install cudnn-cuda-12
export CUDNN_Lib=/usr/lib/x86_64-linux-gnu/


# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet18int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=example-data
export USE_Python=OFF

# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export PYTHONPATH=$BuildDirectory:$PYTHONPATH
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!