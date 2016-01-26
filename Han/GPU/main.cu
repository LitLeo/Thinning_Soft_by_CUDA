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
	Thinning thin_gpu;


	Image *inimg;
    ImageBasicOp::newImage(&inimg);
    int errcode;
    errcode = ImageBasicOp::readFromFile(argv[1], inimg);
    if (errcode != NO_ERROR) {
        cout << "error: " << errcode << endl;
        return 0; 
    }
    for(int i = 0; i < inimg->width * inimg->height; i++)
    {
        if(inimg->imgData[i] != 0)
            inimg->imgData[i] = 255;
    }
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
    for (int i = 0; i < 100; i++) 
       thin_gpu.thinHan(inimg, outimg1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);
	cout << "A2() time is " << (runTime) / 100 << " ms" << endl;
    ImageBasicOp::copyToHost(outimg1);
    ImageBasicOp::writeToFile("A2_outimg.bmp", outimg1); 

    // Pattern 表法，Pattern位于 global 内存
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // float runTime;
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) 
        thin_gpu.thinHanPt(inimg, outimg2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);
    cout << "A3() time is " << (runTime) / 100 << " ms" << endl;
    ImageBasicOp::copyToHost(outimg2);
    ImageBasicOp::writeToFile("A3_outimg.bmp", outimg2); 

    // // Pattern 表法，Pattern位于常量内存
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // for (int i = 0; i < 100; i++) 
    //     thin_gpu.thinHanPtCon(inimg, outimg3);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&runTime, start, stop);
    // cout << "A4() time is " << (runTime) / 100 << " ms" << endl;
    // ImageBasicOp::copyToHost(outimg3);
    // ImageBasicOp::writeToFile("A4_outimg.bmp", outimg3); 

    // // 直接并行,一个线程处理四个点
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // for (int i = 0; i < 100; i++) 
    //     thin_gpu.thinHanFour(inimg, outimg4);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&runTime, start, stop);
    // cout << "A5() time is " << (runTime) / 100 << " ms" << endl;
    // ImageBasicOp::copyToHost(outimg4);
    // ImageBasicOp::writeToFile("A5_outimg.bmp", outimg4); 

    // // Pattern 表法，Pattern位于常量内存,一个线程处理四个点
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // for (int i = 0; i < 100; i++) 
    //     thin_gpu.thinHanPtConFour(inimg, outimg5);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&runTime, start, stop);
    // cout << "A5() time is " << (runTime) / 100 << " ms" << endl;
    // ImageBasicOp::copyToHost(outimg5);
    // ImageBasicOp::writeToFile("A6_outimg.bmp", outimg5); 
    
    ImageBasicOp::deleteImage(inimg);
    ImageBasicOp::deleteImage(outimg1);
    ImageBasicOp::deleteImage(outimg2);
    ImageBasicOp::deleteImage(outimg3);
    ImageBasicOp::deleteImage(outimg4);
    ImageBasicOp::deleteImage(outimg5);
    ImageBasicOp::deleteImage(outimg6);

	return 0;
}
