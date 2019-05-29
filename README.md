# mibileNet_SSD_TensorRT_Robocon2019
2019年robocon全国机器人大赛识别拐骨得分仓库

1. 依赖环境

   mobileNet_ssd

   AlexNet

   nvidia tx2

   cuda9.0

   tensorRT4.0(ARM64 or X64)----关于SSD的实现在Nvidia官网的tensorRT5中已经有实现了

   cuDNN7.4

   glog

   迈德威士的相机

   在3.0版本中把红与蓝数据集混合，在后层中加入AlexNet来判断颜色

   为了加快模型的读取速度，把模型转换为了中间文件。但是程序的首次运行要使用sudo命令。

2.目录结构说明

gflags：一个用来给main函数传参的库。基于这个库，可以把重要的参数提取出来作为入口参数，这样就不要重新编译程序了。

inference_engine：网络推理的主要文件，里面的tensorRT类实现了图片的读取，推理，颜色分类，以及结果的输出。

lib：为了方便此程序在其他程序处调用，把相机，串口，还有推理模块编译成了lib。编译生成的文件在build目录下的lib中。

mindVisionApi：相机的API。注意相机的分辨率在linux下不支持680*480的缩放。分辨率的修改方式见程序。

note：里面的文档***Changes to made for more than 5 classes · Issue #9 · Ghustwb_MobileNet-SSD-TensorRT.pdf***记录了网络结构的修改方式。当网络的输入与输出改变时，可以参照此文档进行修改。

script：程序的运行脚本，因为相机需要sudo运行，所以需要sudo ./

serial：串口程序。需要注意的问题，此串口在收数时会因为，如果主机发数太快，如果从机调用API读取数据的速度不够快，从机会把串口收到的数存起来，导致调用API获得的数据不是实时的。解决办法：a、调用串口中的方法flush来刷新缓存区，但如果传输的数据时连续几个字节时，会导致数据的混乱。b、使用单独的串口多线程。

src：主程序调用。MatCom.hpp中有关于socket的UDP传输图片的实现。tx2作为服务端负责接收客户端发来的图片。关于socket传输可以参考网上的教程以及仓库https://github.com/Yangxiu123321/sockt_opencv.git

tensorRTplugin：网络结构的核心文件，里面用了cuda的核函数来进行加速

util：图片导入cuda的一些接口文件

commit.sh自己写的一个仓库提交脚本。本地需要安装基于ssh的仓库密钥。

3.注意问题

在图像识别时可以加入掩模来实现不必要视野的遮挡