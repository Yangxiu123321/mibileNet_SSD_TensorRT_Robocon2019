#include "main.h"
#include "tensorRT.h"

const char* model  = "../../../model/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "../../../model/MobileNetSSD_deploy.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
//static const uint32_t BATCH_SIZE = 1;

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);
    int playground = 0;
    //serial init
    serial::Serial my_serial("/dev/ttyS0", 115200, serial::Timeout::simpleTimeout(2));
    if(my_serial.isOpen())
    {
        std::cout << "[INFO]" << "serial port initialize ok" << std::endl;
    }else{
        std::cout << "[ERROR]" << "can't find serial" << std::endl;
        return -1;
    }
    // openVINO init
    std::cout << "openvino init\n";
    //TensorRT tensorRT(argc,argv,playground);

    // camera init
    std::cout << "camera init\n";
    MvInit mvCamera(playground/2);

    while(1)
    {
        // get image
	std::cout << "get img\n";
        //tensorRT.srcImg = mvCamera.getImage();

        // inference
	std::cout << "do inference\n";
        //tensorRT.inference();
    }
    return 0;
}
