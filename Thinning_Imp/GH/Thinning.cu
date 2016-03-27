#include "Thinning.h"
#include <iostream>
using namespace std;

static __global__ void _thinGpuIter1Ker(ImageCuda outimg, ImageCuda tempimg, int *devchangecount,
                                            unsigned char highpixel, unsigned char lowpixel)
{
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstc >= tempimg.imgMeta.width - 1 || 
        dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    unsigned char *outptr;
    int curpos = dstr * tempimg.pitchBytes + dstc;
    outptr = tempimg.imgMeta.imgData + curpos;

    if (*outptr != lowpixel) {
        int row1 = (dstr - 1) * tempimg.pitchBytes;
        int row2 = row1 + tempimg.pitchBytes;
        int row3 = row2 + tempimg.pitchBytes;

        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + row1] == highpixel;
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    row1] == highpixel;
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + row1] == highpixel;
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + row2] == highpixel;
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + row3] == highpixel;
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    row3] == highpixel;
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + row3] == highpixel;
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + row2] == highpixel;

        int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                 (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
        int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
        int N  = N1 < N2 ? N1 : N2;
        int m  = ((p6 | p7 | !p1) & p8);

        if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
            outimg.imgMeta.imgData[curpos] = lowpixel;
            *devchangecount = 1;
        } 
    }
}

static __global__ void _thinGpuIter2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount, 
                                            unsigned char highpixel, unsigned char lowpixel)
{
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstc >= tempimg.imgMeta.width - 1 || 
        dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    unsigned char *outptr;
    int curpos = dstr * tempimg.pitchBytes + dstc;
    outptr = tempimg.imgMeta.imgData + curpos;

    if (*outptr != lowpixel) {
        int row1 = (dstr - 1) * tempimg.pitchBytes;
        int row2 = row1 + tempimg.pitchBytes;
        int row3 = row2 + tempimg.pitchBytes;

        unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + row1] == highpixel;
        unsigned char p2 = tempimg.imgMeta.imgData[dstc+    row1] == highpixel;
        unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + row1] == highpixel;
        unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + row2] == highpixel;
        unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + row3] == highpixel;
        unsigned char p6 = tempimg.imgMeta.imgData[dstc+    row3] == highpixel;
        unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + row3] == highpixel;
        unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + row2] == highpixel;

        int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
              (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
        int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
        int N  = N1 < N2 ? N1 : N2;
        int m  = ((p2 | p3 | !p5) & p4);

        if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
            outimg.imgMeta.imgData[curpos] = lowpixel;
            *devchangecount = 1;
        }
    }
}

__host__ int Thinning::thinGpu(Image *inimg, Image *outimg)
{
    int errcode;  
    cudaError_t cudaerrcode; 

    if (inimg == NULL || outimg == NULL)
     return NULL_POINTER;

    int *devchangecount = NULL;
    int changeCount;
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        return CUDA_ERROR;
    }

    Image *tempimg = NULL;
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                             inimg->height);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    changeCount = 1;
    // int iter_num = 0;
    while (changeCount > 0) {
        // iter_num ++;
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                              cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);

        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        _thinGpuIter1Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount,
                                                     highPixel, lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }

        cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);

        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        _thinGpuIter2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                     devchangecount, highPixel, lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }     

        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                              cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

    }
    // cout << "thinGH iter_num = " << iter_num << endl;
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);

    return NO_ERROR;
}

static __global__ void _thinGpuPtIter1Ker(ImageCuda outimg, ImageCuda tempimg, int *devchangecount,
                                            unsigned char highpixel, unsigned char lowpixel)
{
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstc >= tempimg.imgMeta.width - 1 || 
        dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    unsigned char *outptr;
    int curpos = dstr * outimg.pitchBytes + dstc;
    outptr = outimg.imgMeta.imgData + curpos;

    if (*outptr != lowpixel) {

        if(tempimg.imgMeta.imgData[curpos] == lowpixel)
            outimg.imgMeta.imgData[curpos] = lowpixel;
        else {

            int row1 = (dstr - 1) * tempimg.pitchBytes;
            int row2 = row1 + tempimg.pitchBytes;
            int row3 = row2 + tempimg.pitchBytes;

            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + row1] == highpixel;
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    row1] == highpixel;
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + row1] == highpixel;
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + row2] == highpixel;
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + row3] == highpixel;
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    row3] == highpixel;
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + row3] == highpixel;
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + row2] == highpixel;

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
            int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
            int N  = N1 < N2 ? N1 : N2;
            int m  = ((p6 | p7 | !p1) & p8);

            if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
                outimg.imgMeta.imgData[curpos] = lowpixel;
                *devchangecount = 1;
            } 
        }
    }
}

static __global__ void _thinGpuPtIter2Ker(ImageCuda outimg, ImageCuda tempimg, int *devchangecount,
                                            unsigned char highpixel, unsigned char lowpixel)                                   
{
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstc >= tempimg.imgMeta.width - 1 || 
        dstr >= tempimg.imgMeta.height - 1 || dstc < 1 || dstr < 1)
        return;

    unsigned char *outptr;
    int curpos = dstr * outimg.pitchBytes + dstc;
    outptr = outimg.imgMeta.imgData + curpos;

    if (*outptr != lowpixel) {

        if (tempimg.imgMeta.imgData[curpos] == lowpixel)
            outimg.imgMeta.imgData[curpos] = lowpixel;
        else{
            int row1 = (dstr - 1) * tempimg.pitchBytes;
            int row2 = row1 + tempimg.pitchBytes;
            int row3 = row2 + tempimg.pitchBytes;

            unsigned char p1 = tempimg.imgMeta.imgData[dstc-1 + row1] == highpixel;
            unsigned char p2 = tempimg.imgMeta.imgData[dstc+    row1] == highpixel;
            unsigned char p3 = tempimg.imgMeta.imgData[dstc+1 + row1] == highpixel;
            unsigned char p4 = tempimg.imgMeta.imgData[dstc+1 + row2] == highpixel;
            unsigned char p5 = tempimg.imgMeta.imgData[dstc+1 + row3] == highpixel;
            unsigned char p6 = tempimg.imgMeta.imgData[dstc+    row3] == highpixel;
            unsigned char p7 = tempimg.imgMeta.imgData[dstc-1 + row3] == highpixel;
            unsigned char p8 = tempimg.imgMeta.imgData[dstc-1 + row2] == highpixel;

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                  (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
            int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
            int N  = N1 < N2 ? N1 : N2;
            int m  = ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
                outimg.imgMeta.imgData[curpos] = lowpixel;
                *devchangecount = 1;
            }
        }
    }
}

__host__ int Thinning::thinGpuPt(Image *inimg, Image *outimg)
{
    int errcode;  
    cudaError_t cudaerrcode; 

    if (inimg == NULL || outimg == NULL)
     return NULL_POINTER;

    int *devchangecount = NULL;
    int changeCount;
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        return CUDA_ERROR;
    }

    Image *tempimg = NULL;
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                             inimg->height);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    ImageCuda outsubimgCud;
    errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    ImageCuda tempsubimgCud;
    errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
    if (errcode != NO_ERROR) {
        return errcode;
    }

    dim3 gridsize, blocksize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

    cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);

    if (cudaerrcode != cudaSuccess) {
        return CUDA_ERROR;
    }

    int i = 0;

    changeCount = 1;
    // int iter_num = 0;
    while (changeCount > 0) {
        // iter_num ++;
        changeCount = 0;
        cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                              cudaMemcpyHostToDevice);
        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        _thinGpuPtIter1Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount,
                                                    highPixel, lowPixel);
        if (cudaGetLastError() == cudaSuccess) {
            i++;
        }

        else
            return CUDA_ERROR;
            

        _thinGpuPtIter2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount, 
                                                    highPixel, lowPixel);
        if (cudaGetLastError() == cudaSuccess) {
            i++;
        }

        else
            return CUDA_ERROR;

        cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                              cudaMemcpyDeviceToHost);
        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

    }

    if (i % 2 == 0)
        cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                          tempimg->imgData, tempsubimgCud.deviceId, 
                                          tempsubimgCud.pitchBytes * tempimg->height);
    // cout << "thinGH iter_num = " << iter_num << endl;
    cudaFree(devchangecount);
    ImageBasicOp::deleteImage(tempimg);

    return NO_ERROR;
}

