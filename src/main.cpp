#include "main.h"

int main(int argc, char *argv[])
{
    int palyground = 0;
    //serial init
    serial::Serial my_serial(FLAGS_com, 115200, serial::Timeout::simpleTimeout(2));
    if(my_serial.isOpen())
    {
        std::cout << "[INFO]" << "serial port initialize ok" << std::endl;
    }else{
        std::cout << "[ERROR]" << "can't find serial" << std::endl;
        return -1;
    }
    while(1)
    {
        string serialData = my_serial.read(8);
        if(serialData == "action01")
        {
            palyground = 0;
            break;
        }else if(serialData == "action10")
        {
            palyground = 1;
            break;
        }else
        {
            std::cout<<"[STATUS]" << "wait...\n";
        }
    }
    // camera init
    MvInit mvCamera(playground/2);

    // openVINO init
    TensorRT tensorRT(argc,argv,palyground);

    while(1)
    {
        // get image
        tensorRT.srcImg = mvCamera.getImage();
        // inference
        TensorRT.inference();
    }
    return 0;
}
