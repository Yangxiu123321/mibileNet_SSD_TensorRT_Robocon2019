#include "tensorRT.h"
#include <gflags/gflags.h>


/// @brief message for model argument
static const char model_message[] = "Required. Path to an .prototxt file with a trained model.filename no ext";

/// It is a required parameter
DEFINE_string(m_red, "", model_message);

/// It is a required parameter
DEFINE_string(m_blue, "", model_message);

int ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_m_red.empty() || FLAGS_m_blue.empty()) {
        std::cout << "[ERROR]" << "-m not set" << std::endl; 
        //return -1;
    }

    return 0;
}

TensorRT::TensorRT(int argc,char *argv[],int playground)
{
    ParseAndCheckCommandLine(argc,argv);
    playgroundIdx = playground % 2;
    init();
}

TensorRT::~TensorRT()
{
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
}

void TensorRT::init(void)
{
    std::string modelName = FLAGS_m_red + ".prototxt";
    std::string weightName = FLAGS_m_red + ".caffemodel";
    if(playgroundIdx)
    {
        modelName = FLAGS_m_red + ".prototxt";
        weightName = FLAGS_m_red + ".caffemodel";
    }else
    {
        modelName = FLAGS_m_blue + ".prototxt";
        weightName = FLAGS_m_blue + ".caffemodel";
    }
    
    std::cout << "modelname:" << modelName << "\n" << "weightName:" << weightName << std::endl;

    //const char* model = modelName.data();
    //const char* weight = weightName.data();
    //const char* model = "/home/nvidia/code/tensorRT/mibileNet_SSD_TensorRT_Robocon2019/model/red/MobileNetSSD_deploy.prototxt";
    //const char* weight = "/home/nvidia/code/tensorRT/mibileNet_SSD_TensorRT_Robocon2019/model/red/MobileNetSSD_deploy.caffemodel";
    const char* weight  = "../../../model/MobileNetSSD_deploy.caffemodel";
    const char* model = "../../../model/MobileNetSSD_deploy_iplugin.prototxt";
    
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);
    std::cout << "load model finish\n";

    dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    dimsOut = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    data = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    output = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
}

float* TensorRT::allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}

void TensorRT::loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

bool TensorRT::inference(void)
{
    if(srcImg.empty())
    {
        std::cout << "no imageData" << std::endl;
        return false;
    }
    cv::Mat frame;
    srcImg.copyTo(frame);

    cv::resize(frame, frame, cv::Size(300,300));
    const size_t size = IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float3);

    if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
    {
        cout <<"Cuda Memory allocation error occured."<<endl;
        return false;
    }

    void* imgData = malloc(size);
    memset(imgData,0,size);

    loadImg(frame,IMAGE_HEIGHT,IMAGE_WIDTH,(float*)imgData,make_float3(127.5,127.5,127.5),0.007843);
    cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

    void* buffers[] = { imgCUDA, output };

    tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);

    vector<vector<float> > detections;

    for (int k=0; k<3; k++)
    {
        if(output[7*k+1] == -1)
            break;
    float confidence = output[7*k+2];
        
    if(confidence < 0.6)
    {
        continue;
    }
    float classIndex = output[7*k+1];
    float xmin = output[7*k + 3];
    float ymin = output[7*k + 4];
    float xmax = output[7*k + 5];
    float ymax = output[7*k + 6];
    std::cout << classIndex << " , " << confidence << " , "  << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
    int x1 = static_cast<int>(xmin * srcImg.cols);
    int y1 = static_cast<int>(ymin * srcImg.rows);
    int x2 = static_cast<int>(xmax * srcImg.cols);
    int y2 = static_cast<int>(ymax * srcImg.rows);
    cv::rectangle(srcImg,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,255),1);
    }
    //cv::imshow("mobileNet",srcImg);
    free(imgData);
    return true;
}
