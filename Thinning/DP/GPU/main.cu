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
	Thinning thin;


	Image *inimg;
    ImageBasicOp::newImage(&inimg);
    int errcode;
    errcode = ImageBasicOp::readFromFile(argv[1], inimg);
    if (errcode != NO_ERROR) {
        cout << "error: " << errcode << endl;
        return 0; 
    }

    for(int i = 0; i < inimg->width * inimg->height; i++)
    if(inimg->imgData[i] != 0)
        inimg->imgData[i] = 255;

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
    
    thin.thinDP(inimg, outimg1);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);
	cout << "thinDP() time is " << runTime << " ms" << endl;
    ImageBasicOp::copyToHost(outimg1);
    ImageBasicOp::writeToFile("thinDP_outimg.bmp", outimg1); 

    ImageBasicOp::deleteImage(inimg);
    ImageBasicOp::deleteImage(outimg1);

	return 0;
}
