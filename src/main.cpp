#include "main.h"
#include "tensorRT.h"

int main(int argc, char *argv[])
{
    int playground = 1;
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
    // playground-1 only want to be test
    MvInit mvCamera(playground-1);
    //tensorRT.srcImg =cv::imread("/home/nvidia/code/tensorRT/mibileNet_SSD_TensorRT_Robocon2019/test_picture/1.BMP");
    while(1)
    {
        // get image
	//std::cout << "get img\n";
        //cv::Mat srcImg = mvCamera.getImage();
	//cv::flip(srcImg,tensorRT.srcImg,1);
        tensorRT.srcImg = mvCamera.getImage();
        //cap >> tensorRT.srcImg;
	cv::imshow("src",tensorRT.srcImg);

        // inference
	std::cout << "do inference\n";
        tensorRT.inference();
        //cv::imshow("src",tensorRT.debugImg);
	
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
