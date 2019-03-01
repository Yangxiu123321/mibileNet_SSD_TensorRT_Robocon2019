#!/bin/bash

cd ../build/build/bin

./mobileNet -m_red=../model/red/MobileNetSSD_deploy -m_blue=../model/blue/MobileNetSSD_deploy -com=/dev/ttyS0
