#!/bin/bash

cd ../build/build/bin

./mobileNet -m_red=/home/nvidia/code/tensorRT/mibileNet_SSD_TensorRT_Robocon2019/model/red/MobileNetSSD_deploy -m_blue=/home/nvidia/code/tensorRT/mibileNet_SSD_TensorRT_Robocon2019/model/blue/MobileNetSSD_deploy -com=/dev/ttyS0
