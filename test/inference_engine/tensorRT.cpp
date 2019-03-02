#include "tensorRT.h"
#include <gflags/gflags.h>


const char* weight  = "../../../blue/MobileNetSSD_deploy.caffemodel";
const char* model = "../../../blue/MobileNetSSD_deploy.prototxt";


const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";

static const uint32_t BATCH_SIZE = 1;



TensorRT::TensorRT(int argc,char *argv[],int playground)
{

}

TensorRT::~TensorRT()
{
    //cudaFree(imgCUDA);
    //cudaFreeHost(imgCPU);
    //cudaFree(output);
    //tensorNet.destroy();
}


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
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


void TensorRT::init(void)
{
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);
    std::cout << "load model finish\n";

    dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    dimsOut = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    data = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    output = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
}


int TensorRT::inference(void)
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    int height = 300;
    int width  = 300;

    cv::Mat frame,srcImg;

    void* imgCPU;
    void* imgCUDA;

//    std::string imgFile = "../../testPic/test.jpg";
//    frame = cv::imread(imgFile);
    cv::VideoCapture cap("/home/nvidia/Videos/11.avi");
    while(1)
    {

        cap >> frame;

        if(frame.empty())
        {
            std::cout << "no imageData" << std::endl;
            break;
        }

        srcImg = frame.clone();

        cv::resize(frame, frame, cv::Size(300,300));
        const size_t size = width * height * sizeof(float3);

        if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }

        void* imgData = malloc(size);
        memset(imgData,0,size);

        loadImg(frame,height,width,(float*)imgData,make_float3(127.5,127.5,127.5),0.007843);
        cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

        void* buffers[] = { imgCUDA, output };

        tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);

        vector<vector<float> > detections;

        for (int k=0; k<3; k++)
        {
            if(output[7*k+1] == -1)
                break;
            float classIndex = output[7*k+1];
            float confidence = output[7*k+2];
            if(confidence < 0.7)
		{
		   continue;
		}
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
        cv::imshow("mobileNet",srcImg);
        int c = cv::waitKey(30);
        if(c == 27 || c == 'q' || c == 'Q')
        {
            break;
        }
        free(imgData);
    }
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
