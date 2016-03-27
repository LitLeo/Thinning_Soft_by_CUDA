#include <iostream>
#include "Thinning.h"
#include "ErrorCode.h"
#include "Image.h"
using namespace std;

int main(int argc, char const **argv)
{
	if(argc < 2)
	{
		cout << "Please input image!" << endl;
		return 0;
	}

	Image *inimg;
    ImageBasicOp::newImage(&inimg);
    int errcode;
    errcode = ImageBasicOp::readFromFile(argv[1], inimg);
    if (errcode != NO_ERROR) {
        cout << "error: " << errcode << endl;
        return 0; 
    }
    cout << "image: " << argv[1] << endl;
    for(int i = 0; i < inimg->width * inimg->height; i++)
    if(inimg->imgData[i] != 0)
        inimg->imgData[i] = 255;

    // get the device count
    int deviceCount = 0;
    cudaError_t error_id ;
    error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        cout << "error: " << errcode << endl;
        return 0; 
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cout << "Device " << dev << " " << deviceProp.name << endl; // , dev, deviceProp.name);

        for (int by = 0; by <= 32; by += 2) {
            Thinning thin_gpu;
            if (by == 0)
                thin_gpu.DEF_BLOCK_Y = 1;
            else
                thin_gpu.DEF_BLOCK_Y = by;

            cout << "\nDEF_BLOCK_Y = " << thin_gpu.DEF_BLOCK_Y << " DEF_BLOCK_X = " << thin_gpu.DEF_BLOCK_X << endl;

            Image *outimg1;
            ImageBasicOp::newImage(&outimg1);
            ImageBasicOp::makeAtHost(outimg1, inimg->width, inimg->height);

            cudaEvent_t start, stop;
            float runTime;

            warmup();
            // 直接并行
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            
            thin_gpu.thinGH(inimg, outimg1);
        
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runTime, start, stop);
            cout << "thinGH() time is " << runTime<< " ms" << endl;
            ImageBasicOp::copyToHost(outimg1);
            ImageBasicOp::writeToFile("thinGH_outimg.bmp", outimg1); 

            ImageBasicOp::deleteImage(outimg1);
        }
    }
    cudaDeviceSynchronize();
    cudaDeviceReset();
    ImageBasicOp::deleteImage(inimg);
            
	return 0;
}
