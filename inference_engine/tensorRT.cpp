#include "tensorRT.h"
#include <gflags/gflags.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .prototxt file with a trained model.filename no ext";

/// It is a required parameter
DEFINE_string(m_red, "", model_message);

/// It is a required parameter
DEFINE_string(m_blue, "", model_message);

/// It is a required parameter
DEFINE_string(m_alex, "", model_message);

/// It is a required parameter
DEFINE_string(m_mean, "", model_message);

int ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_m_red.empty() || FLAGS_m_blue.empty() || FLAGS_m_mean.empty()) {
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
    //cudaFree(imgCUDA);
    //cudaFreeHost(imgCPU);
    //cudaFree(output);
    //tensorNet.destroy();
}

void TensorRT::init(void)
{
    std::string meanfile = FLAGS_m_mean;
    const char* mean = meanfile.data();
    // Parse mean file
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(mean);
    parser->destroy();

    // Subtract mean from image
    std::cout << "start import meanfile...\n";
    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());
    DimsNCHW testDim = meanBlob->getDimensions();
    std::cout << "meanfile(nchw):" << testDim.n() <<" " << testDim.c() << " " << testDim.h() << " " << testDim.w() << std::endl;

    for (int channel = 0; channel < 3; ++channel) {
        int pixels = 227 * 227;
        for (int i = 0; i < pixels; ++i) {
        meanDataBGR[channel * pixels + i] = float(meanData[i * 3 + 2 - channel]);
        }
    }
    meanBlob->destroy();\
    std::cout << "import meanfile sucessfully\n";

    std::string modelName = FLAGS_m_red + ".prototxt";
    std::string weightName = FLAGS_m_red + ".caffemodel";
    std::string modelName2 = FLAGS_m_alex + ".prototxt";
    std::string weightName2 = FLAGS_m_alex + ".caffemodel";
    if(playgroundIdx)
    {
        modelName = FLAGS_m_blue + ".prototxt";
        weightName = FLAGS_m_blue + ".caffemodel";
    }else
    {
        modelName = FLAGS_m_red + ".prototxt";
        weightName = FLAGS_m_red + ".caffemodel";
    }
    
    std::cout << "modelname:" << modelName << "\n" << "weightName:" << weightName << std::endl;

    const char* model = modelName.data();
    const char* weight = weightName.data();
    const char* model2 = modelName2.data();
    const char* weight2 = weightName2.data();
    //const char* weight  = "../../../model/MobileNetSSD_deploy.caffemodel";
    //const char* model = "../../../model/MobileNetSSD_deploy_iplugin.prototxt";
    
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,
                         model2,weight2,INPUT_BLOB_NAME2, output_vector2,
                         BATCH_SIZE);
    std::cout << "load model finish\n";

    dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    dimsOut = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
    dimsData2 = tensorNet.getTensorDimsForAlex(INPUT_BLOB_NAME2);
    dimsOut2 = tensorNet.getTensorDimsForAlex(OUTPUT_BLOB_NAME2);

    data = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    output = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    data2 = allocateMemory( dimsData2 , (char*)"input2 blob");
    std::cout << "allocate data2" << std::endl;
    output2 = allocateMemory( dimsOut2  , (char*)"output2 blob");
    std::cout << "allocate output2" << std::endl;
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
    srcImg.copyTo(debugImg);
    cv::resize(frame, frame, cv::Size(300,300));
    cv::resize(debugImg, debugImg, cv::Size(640,480));
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

    std::cout << "start global inference\n";
    tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);
    std::cout << "end global inference\n";

    vector<vector<float> > detections;
    
    int boneScore = 0;
    for (int k=0; k<3; k++)
    {
        if(output[7*k+1] == -1)
            break;
        int classIndex = output[7*k+1];
        if(classIndex == 0 || classIndex == 4)
        {
        continue;
        }
        float confidence = output[7*k+2];   
        if(confidence < 0.6)
        {
            continue;
        }
        float xmin = output[7*k + 3];
        float ymin = output[7*k + 4];
        float xmax = output[7*k + 5];
        float ymax = output[7*k + 6];
        int x1 = static_cast<int>(xmin * debugImg.cols);
        int y1 = static_cast<int>(ymin * debugImg.rows);
        int x2 = static_cast<int>(xmax * debugImg.cols);
        int y2 = static_cast<int>(ymax * debugImg.rows);
        std::cout << classIndex << " , " << confidence << std::endl;
        int weidth = x2 - x1;
        int height = y2 - y1;
        cv::Mat roiImg;
        debugImg(cv::Rect(x1,y1,weidth,height)).copyTo(roiImg);
        cv::resize(roiImg,roiImg,cv::Size(227,227));
        imshow("roi",roiImg);
        for (int channel = 0; channel < 3; ++channel)
        {
            int pixels = 227 * 227;
            for (int i = 0; i < pixels; ++i)
            {
                roiData[channel * pixels + i] = float(roiImg.data[i * 3 + channel]);
            }
        }
        const size_t roiSize = 227 * 227 * sizeof(float3);
        if( CUDA_FAILED( cudaMalloc( &roiCUDA, roiSize)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }
        float* roiDataBGR = (float*)malloc(roiSize);
        memset(roiDataBGR,0,roiSize);
        for(int i = 0;i<227*227*3;i++)
        {
            roiDataBGR[i] = roiData[i] - meanDataBGR[i];
        }
        if(CUDA_FAILED( cudaMemcpyAsync(roiCUDA,roiDataBGR,roiSize,cudaMemcpyHostToDevice)))
        {
            cout <<"Cuda trancefor data error occured."<<endl;
            return false;
        }

        void* buffers2[] = { roiCUDA, output2 };
        std::cout << "start roi inference\n";
        //std::cout << "roi size:" << roiSize << "\n";
        int continueFlag = 0;
        tensorNet.imageInferenceForAlex( buffers2, output_vector2.size() + 1, BATCH_SIZE,&continueFlag,playgroundIdx);
        if(continueFlag)
        {
            std::cout << "finish continue\n";
            free(roiDataBGR);
            continue;
        }
        switch(int(classIndex))
        {
            case 1:
                boneScore += 50;
            break;
            case 2:
                boneScore += 40;
            break;
            case 3:
                boneScore += 20;
            break;
            default:
            break;
        }
        if(boneScore > 49)
        {
            runFlag = 1;
        }else
        {
            runFlag = 0;
        }
        std::cout << "\033[31mboneScore:\033[0m" << boneScore << "\n" << "runFlag:" << runFlag << std::endl;
        std::cout << "end roi inference\n";
        cv::rectangle(debugImg,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,255),1);
        cv::imshow("mobileNet",debugImg);
        free(roiDataBGR);
    }
    free(imgData);
    return true;
}

void TensorRT::freeTensor(void)
{
    cudaFree(imgCUDA);
    cudaFree(roiCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    cudaFree(output2);
    tensorNet.destroy();
}



