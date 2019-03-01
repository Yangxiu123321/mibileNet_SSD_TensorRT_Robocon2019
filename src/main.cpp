#include "main.h"
#include "tensorRT.h"

int main(int argc, char *argv[])
{
    int playground = 0;
    //serial init
    serial::Serial my_serial(FLAGS_com, 115200, serial::Timeout::simpleTimeout(2));
    if(my_serial.isOpen())
    {
        std::cout << "[INFO]" << "serial port initialize ok" << std::endl;
    }else{
        std::cout << "[ERROR]" << "can't find serial" << std::endl;
        return -1;
    }
    // camera init
    MvInit mvCamera(playground/2);

    // openVINO init
    TensorRT tensorRT(argc,argv,playground);

    while(1)
    {
        // get image
        tensorRT.srcImg = mvCamera.getImage();
        // inference
        tensorRT.inference();
    }
    return 0;
}
