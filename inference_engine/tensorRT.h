#ifndef __TENSORRT_H
#define __TENSORRT_H

#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "imageBuffer.h"
#include <opencv2/opencv.hpp>



#define IMAGE_HEIGHT 300
#define IMAGE_WIDTH 300

#define BATCH_SIZE 1

class TensorRT
{
public:
    TensorRT(int argc,char *argv[],int playground);
    ~TensorRT();
    float* allocateMemory(DimsCHW dims, char* info);
    void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale );

    void init(void);
    bool inference(void);
    void freeTensor(void);

    cv::Mat srcImg;
    cv::Mat debugImg;

    int playgroundIdx;

    int runFlag = 0;
    //第一个快不是50分时，置此标志位
    int breakFlag = 0;
    int runTimeLimit = 20;
    int boundaryErrorLimit = 10;
    int isShowDebugImg = 1;
    int isUseAlex = 1;
    float boneConfidence = 0.6;
  
private:
    TensorNet tensorNet;
	/* data */
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "detection_out";
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};

    DimsCHW dimsData;
    DimsCHW dimsOut;

    float* data = NULL;
    float* output = NULL;

    //void* imgCPU;
    void* imgCUDA;

    /*data2*/
    const char* INPUT_BLOB_NAME2 = "data";
    const char* OUTPUT_BLOB_NAME2 = "prob";
    std::vector<std::string> output_vector2 = {OUTPUT_BLOB_NAME2};
    DimsCHW dimsData2;
    DimsCHW dimsOut2;

    float* data2 = NULL;
    float* output2 = NULL;

    // bone score
    int boneScoreNum_50 = 0;
    int boneScoreNum_40 = 0;
    int boneScoreNum_20 = 0;
    // coordition stored
    int x1Last = 0;
    int y1Last = 0;
    int x2Last = 0;
    int y2Last = 0;

    float *roiData = new float[227*227*3];

    void* roiCUDA;

    // mean file
    float *meanDataBGR = new float[227 * 227 * 3];
};

#endif