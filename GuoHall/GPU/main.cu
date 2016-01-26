#include <iostream>
#include "Thinning.h"
#include "ErrorCode.h"
#include "Image.h"
using namespace std;

#define LOOP 1000

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

            Image *outimg2;
            ImageBasicOp::newImage(&outimg2);
            ImageBasicOp::makeAtHost(outimg2, inimg->width, inimg->height);

            Image *outimg3;
            ImageBasicOp::newImage(&outimg3);
            ImageBasicOp::makeAtHost(outimg3, inimg->width, inimg->height);

            Image *outimg4;
            ImageBasicOp::newImage(&outimg4);
            ImageBasicOp::makeAtHost(outimg4, inimg->width, inimg->height);

            Image *outimg5;
            ImageBasicOp::newImage(&outimg5);
            ImageBasicOp::makeAtHost(outimg5, inimg->width, inimg->height);

            Image *outimg6;
            ImageBasicOp::newImage(&outimg6);
            ImageBasicOp::makeAtHost(outimg6, inimg->width, inimg->height);
            
            cudaEvent_t start, stop;
            float runTime;

            // 直接并行
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            for (int i = 0; i < LOOP; i++) 
                thin_gpu.thinGpu(inimg, outimg1);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runTime, start, stop);
            cout << "thinGpu() time is " << (runTime) / LOOP << " ms" << endl;
            ImageBasicOp::copyToHost(outimg1);
            ImageBasicOp::writeToFile("thinGpu_outimg.bmp", outimg1); 

            // Pattern 表法，Pattern位于 global 内存
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            // float runTime;
            cudaEventRecord(start, 0);
            for (int i = 0; i < LOOP; i++) 
                thin_gpu.thinGpuPt(inimg, outimg2);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runTime, start, stop);
            cout << "thinGpuPt() time is " << (runTime) / LOOP << " ms" << endl;
            ImageBasicOp::copyToHost(outimg2);
            ImageBasicOp::writeToFile("thinGpuPt_outimg.bmp", outimg2); 

            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);
            // cudaEventRecord(start, 0);
            // for (int i = 0; i < LOOP; i++) 
            //     thin_gpu.thinGpuFour(inimg, outimg5);
            // cudaEventRecord(stop, 0);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&runTime, start, stop);
            // cout << "thinGpuFour() time is " << (runTime) / LOOP << " ms" << endl;
            // ImageBasicOp::copyToHost(outimg5);
            // ImageBasicOp::writeToFile("thinGpuFour_outimg.bmp", outimg5); 

            // // 直接并行,一个线程处理四个点
            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);
            // cudaEventRecord(start, 0);
            // for (int i = 0; i < LOOP; i++) 
            //     thin_gpu.thinGpuPtFour(inimg, outimg4);
            // cudaEventRecord(stop, 0);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&runTime, start, stop);
            // cout << "thinGpuPtFour() time is " << (runTime) / LOOP << " ms" << endl;
            // ImageBasicOp::copyToHost(outimg4);
            // ImageBasicOp::writeToFile("thinGpuPtFour_outimg.bmp", outimg4); 
            
            ImageBasicOp::deleteImage(outimg1);
            ImageBasicOp::deleteImage(outimg2);
            ImageBasicOp::deleteImage(outimg3);
            ImageBasicOp::deleteImage(outimg4);
            ImageBasicOp::deleteImage(outimg5);
            ImageBasicOp::deleteImage(outimg6);

        }
    }
    cudaDeviceSynchronize();
    cudaDeviceReset();
    ImageBasicOp::deleteImage(inimg);
            
	return 0;
}
