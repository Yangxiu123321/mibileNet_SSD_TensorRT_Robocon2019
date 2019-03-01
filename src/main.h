#ifndef __MAIN_H

#define __MAIN_H


#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>

// serial
#include <cstdio>
#include <unistd.h>
#include "serial.h"
#include <mv_init.h>
#include "tensorRT.h"
#include <gflags/gflags.h>

/// @brief message for serial argument
const char com_message[] = "com meaage";

/// It is a required parameter
DEFINE_string(com,"/dev/ttyS0", com_message);

#endif