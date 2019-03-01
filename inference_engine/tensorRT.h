#ifndef __TENSORRT_H

#include "opencv2/opencv.hpp"

#define __TENSORRT_H

#define IMAGE_HEIGHT 480
#define IMAGE_WIDTH 640

#define BATCH_SIZE 1

class TensorRT
{
public:
    TensorRT(/* args */);
    ~TensorRT();
    float* allocateMemory(DimsCHW dims, char* info);
    void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale );

    void init(void);
    void inference(void);

    cv::Mat srcImg;

    int playgroundIdx;
private:
    /* data */
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "detection_out";
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    DimsCHW dimsData;
    DimsCHW dimsOut;

    float* data;
    float* output;

    void* imgCPU;
    void* imgCUDA;
};



#endif
