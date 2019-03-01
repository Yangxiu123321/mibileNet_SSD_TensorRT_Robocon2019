#include "main.h"
#include "tensorRT.h"

int main(int argc, char *argv[])
{
    int playground = 0;
    //serial init
    std::cout << "[INFO]" << "serial init...\n";
    serial::Serial my_serial(FLAGS_com, 115200, serial::Timeout::simpleTimeout(2));
    if(my_serial.isOpen())
    {
        std::cout << "[INFO]" << "serial port initialize ok" << std::endl;
    }else{
        std::cout << "[ERROR]" << "can't find serial" << std::endl;
        return -1;
    }
    // openVINO init
    std::cout << "openvino init\n";
    TensorRT tensorRT(argc,argv,playground);

    // camera init
    std::cout << "camera init\n";
    MvInit mvCamera(playground);

    while(1)
    {
        // get image
	//std::cout << "get img\n";
        tensorRT.srcImg = mvCamera.getImage();
	cv::imshow("src",tensorRT.srcImg);

        // inference
	//std::cout << "do inference\n";
        tensorRT.inference();
	
        int c = cv::waitKey(1);
	if(c == 27 || c == 'q' || c == 'Q')
	{
	  std::cout << "[INFO]" << "finish !!!\n";
	  break;
	}
    }
    std::cout << "[INFO]" << "free tensor memory!!!\n";
    tensorRT.freeTensor();
    return 0;
}
