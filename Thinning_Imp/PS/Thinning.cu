// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
#include <stdio.h>
using namespace std;

#define uchar unsigned char

#define HIGH 255
#define LOW 0

static __global__ void _thinPet1Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;
        int posColumn4 = posColumn3 + tempimg.pitchBytes;

        uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
        uchar x4 = tempimg.imgMeta.imgData[posColumn1 - 1 + dstc] == HIGH;
        uchar x5 = tempimg.imgMeta.imgData[posColumn2 - 1 + dstc] == HIGH;
        uchar x6 = tempimg.imgMeta.imgData[posColumn3 - 1 + dstc] == HIGH;
        uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
        uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
        // uchar y1 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        uchar y2 = tempimg.imgMeta.imgData[posColumn4     + dstc] == HIGH;
        // uchar y3 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
        // uchar y4 = tempimg.imgMeta.imgData[posColumn1 + 2 + dstc] == HIGH;
        uchar y5 = tempimg.imgMeta.imgData[posColumn2 + 2 + dstc] == HIGH;
        // uchar y6 = tempimg.imgMeta.imgData[posColumn3 + 2 + dstc] == HIGH;

        int A  = (x2 ^ x3) + (x3 ^ x4) + (x4 ^ x5) + (x5 ^ x6) + 
                 (x6 ^ x7) + (x7 ^ x8) + (x8 ^ x1) + (x1 ^ x2);
        int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
        int R = x1 && x7 && x8 &&
               ((!y5 && x2 && x3 && !x5) || (!y2 && !x3 && x5 && x6));
        if (A == 2 && B >= 2 && B <= 6 && R == 0) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

static __global__ void _thinPet2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * tempimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = tempimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
        // 防止下面细化处理时重复计算。
        int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
        int posColumn2 = posColumn1 + tempimg.pitchBytes;
        int posColumn3 = posColumn2 + tempimg.pitchBytes;
        // int posColumn4 = posColumn3 + tempimg.pitchBytes;

        uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
        uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
        uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
        uchar x4 = tempimg.imgMeta.imgData[posColumn1 - 1 + dstc] == HIGH;
        uchar x5 = tempimg.imgMeta.imgData[posColumn2 - 1 + dstc] == HIGH;
        uchar x6 = tempimg.imgMeta.imgData[posColumn3 - 1 + dstc] == HIGH;
        uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
        uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;

        int S0 = (x3&&x7) || (x5&&x1);
        int S1 = (x1 && !x6 && (!x4 || x3)) || (x3 && !x8 && (!x6 || x5)) ||
            (x7 && !x4 && (!x2 || x1)) || (x5 && !x2 && (!x8 || x7));
        int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
        int R = (x3 && (x1&&!x8 || x5&&!x6)) || (x7 && (!x5&&!x8 || !x1&&!x6));
        if ((!S0 && S1) && R == 0 && B >= 3) {
            outimg.imgMeta.imgData[curpos] = LOW;
            // 记录删除点数的 devchangecount 值加 1 。
            *devchangecount = 1;
        }
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinPet(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
     int errcode;  
     cudaError_t cudaerrcode; 

     // 检查输入图像，输出图像是否为空。
     if (inimg == NULL || outimg == NULL)
         return NULL_POINTER;

     // 声明所有中间变量并初始化为空。
     Image *tempimg = NULL;
     int *devchangecount = NULL;

     // 记录细化点数的变量，位于 host 端。
     int changeCount;

     // 记录细化点数的变量，位于 device 端。并为其申请空间。
     cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
         return errcode;
     }

     // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
     // device 端。
     errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }
     
     // 提取输出图像
     ImageCuda outsubimgCud;
     errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }

     // 提取暂存图像
     ImageCuda tempsubimgCud;
     errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }

     // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
     dim3 gridsize, blocksize;
     blocksize.x = DEF_BLOCK_X;
     blocksize.y = DEF_BLOCK_Y;
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

     // 赋值为 1，以便开始第一次迭代。
     changeCount = 1;

     // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
     // 停止迭代。 
     while (changeCount > 0) {
         // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
         changeCount = 0;
         cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                  cudaMemcpyHostToDevice);
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPet1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

         // copy ouimg to tempimg 
         cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                      outimg->imgData, outsubimgCud.deviceId, 
                                      outsubimgCud.pitchBytes * outimg->height);
        
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第二步细化操作。
         _thinPet2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() != cudaSuccess) {
             // 核函数出错，结束迭代函数，释放申请的变量空间 。
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }     
        
         // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
         // 变量，进行迭代判断。
         cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                  cudaMemcpyDeviceToHost);
         if (cudaerrcode != cudaSuccess) {
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

    }
     // 细化结束后释放申请的变量空间。
     cudaFree(devchangecount);
     ImageBasicOp::deleteImage(tempimg);
     return NO_ERROR;
}

static __global__ void _thinPetPt1Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * outimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        if(tempimg.imgMeta.imgData[curpos] == LOW)
            outimg.imgMeta.imgData[curpos] = LOW;

        else {

            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;
            int posColumn4 = posColumn3 + tempimg.pitchBytes;

            uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
            uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
            uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
            uchar x4 = tempimg.imgMeta.imgData[posColumn1 - 1 + dstc] == HIGH;
            uchar x5 = tempimg.imgMeta.imgData[posColumn2 - 1 + dstc] == HIGH;
            uchar x6 = tempimg.imgMeta.imgData[posColumn3 - 1 + dstc] == HIGH;
            uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
            uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;
            // uchar y1 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
            uchar y2 = tempimg.imgMeta.imgData[posColumn4     + dstc] == HIGH;
            // uchar y3 = tempimg.imgMeta.imgData[posColumn4 + 1 + dstc] == HIGH;
            // uchar y4 = tempimg.imgMeta.imgData[posColumn1 + 2 + dstc] == HIGH;
            uchar y5 = tempimg.imgMeta.imgData[posColumn2 + 2 + dstc] == HIGH;
            // uchar y6 = tempimg.imgMeta.imgData[posColumn3 + 2 + dstc] == HIGH;

            int A  = (x2 ^ x3) + (x3 ^ x4) + (x4 ^ x5) + (x5 ^ x6) + 
                     (x6 ^ x7) + (x7 ^ x8) + (x8 ^ x1) + (x1 ^ x2);
            int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
            int R = x1 && x7 && x8 &&
                   ((!y5 && x2 && x3 && !x5) || (!y2 && !x3 && x5 && x6));
            if (A == 2 && B >= 2 && B <= 6 && R == 0) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

static __global__ void _thinPetPt2Ker(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // dstc 和 dstr 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int dstc = blockIdx.x * blockDim.x + threadIdx.x;
    int dstr = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    if (dstc >= tempimg.imgMeta.width - 2 || 
     dstr >= tempimg.imgMeta.height - 2 || dstc < 2 || dstr < 2)
     return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    int curpos = dstr * outimg.pitchBytes + dstc;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (*outptr != LOW) {
        if(tempimg.imgMeta.imgData[curpos] == LOW)
            outimg.imgMeta.imgData[curpos] = LOW;

        else {
            // 由于图像是线性存储的，所以在这里先获得 8 邻域里三列的列索引值，
            // 防止下面细化处理时重复计算。
            int posColumn1 = (dstr - 1) * tempimg.pitchBytes;
            int posColumn2 = posColumn1 + tempimg.pitchBytes;
            int posColumn3 = posColumn2 + tempimg.pitchBytes;
            // int posColumn4 = posColumn3 + tempimg.pitchBytes;

            uchar x1 = tempimg.imgMeta.imgData[posColumn2 + 1 + dstc] == HIGH;
            uchar x2 = tempimg.imgMeta.imgData[posColumn1 + 1 + dstc] == HIGH;
            uchar x3 = tempimg.imgMeta.imgData[posColumn1     + dstc] == HIGH;
            uchar x4 = tempimg.imgMeta.imgData[posColumn1 - 1 + dstc] == HIGH;
            uchar x5 = tempimg.imgMeta.imgData[posColumn2 - 1 + dstc] == HIGH;
            uchar x6 = tempimg.imgMeta.imgData[posColumn3 - 1 + dstc] == HIGH;
            uchar x7 = tempimg.imgMeta.imgData[posColumn3     + dstc] == HIGH;
            uchar x8 = tempimg.imgMeta.imgData[posColumn3 + 1 + dstc] == HIGH;

            int S0 = (x3&&x7) || (x5&&x1);
            int S1 = (x1 && !x6 && (!x4 || x3)) || (x3 && !x8 && (!x6 || x5)) ||
                (x7 && !x4 && (!x2 || x1)) || (x5 && !x2 && (!x8 || x7));
            int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
            int R = (x3 && (x1&&!x8 || x5&&!x6)) || (x7 && (!x5&&!x8 || !x1&&!x6));
            if ((!S0 && S1) && R == 0 && B >= 3) {
                outimg.imgMeta.imgData[curpos] = LOW;
                // 记录删除点数的 devchangecount 值加 1 。
                *devchangecount = 1;
            }
        }
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinPetPt(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
     int errcode;  
     cudaError_t cudaerrcode; 

     // 检查输入图像，输出图像是否为空。
     if (inimg == NULL || outimg == NULL)
         return NULL_POINTER;

     // 声明所有中间变量并初始化为空。
     Image *tempimg = NULL;
     int *devchangecount = NULL;

     // 记录细化点数的变量，位于 host 端。
     int changeCount;

     // 记录细化点数的变量，位于 device 端。并为其申请空间。
     cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
     if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
     }

     // 生成暂存图像。
     errcode = ImageBasicOp::newImage(&tempimg);
     if (errcode != NO_ERROR)
         return errcode;
     errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                 inimg->height);
     if (errcode != NO_ERROR) {
         return errcode;
     }

     // 将输入图像 inimg 完全拷贝到输出图像 outimg ，并将 outimg 拷贝到 
     // device 端。
     errcode = ImageBasicOp::copyToCurrentDevice(inimg, outimg);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }
     
     // 提取输出图像
     ImageCuda outsubimgCud;
     errcode = ImageBasicOp::roiSubImage(outimg, &outsubimgCud);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }

     // 提取暂存图像
     ImageCuda tempsubimgCud;
     errcode = ImageBasicOp::roiSubImage(tempimg, &tempsubimgCud);
     if (errcode != NO_ERROR) {
         // FAIL_THIN_IMAGE_FREE;
         return errcode;
     }

     // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量。
     dim3 gridsize, blocksize;
     blocksize.x = DEF_BLOCK_X;
     blocksize.y = DEF_BLOCK_Y;
     gridsize.x = (outsubimgCud.imgMeta.width + blocksize.x - 1) / blocksize.x;
     gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;

     // 赋值为 1，以便开始第一次迭代。
     changeCount = 1;

     // copy ouimg to tempimg 
    cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);
        
    if (cudaerrcode != cudaSuccess) {
        return CUDA_ERROR;
    }

    int i = 0;
     // 开始迭代，当不可再被细化，即记录细化点数的变量 changeCount 的值为 0 时，
     // 停止迭代。 
     while (changeCount > 0) {
         // 将 host 端的变量赋值为 0 ，并将值拷贝到 device 端的 devchangecount。
         changeCount = 0;
         cudaerrcode = cudaMemcpy(devchangecount, &changeCount, sizeof (int),
                                  cudaMemcpyHostToDevice);
         if (cudaerrcode != cudaSuccess) {
             return CUDA_ERROR;
         }

         // 调用核函数，开始第一步细化操作。
         _thinPetPt1Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
         if (cudaGetLastError() == cudaSuccess) {
             i++;
         }
         else
            return CUDA_ERROR;

         
         // 调用核函数，开始第二步细化操作。
         _thinPetPt2Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount);
         if (cudaGetLastError() == cudaSuccess) {
             i++;
         }     
        
        else
            return CUDA_ERROR;
         // 将位于 device 端的 devchangecount 拷贝到 host 端上的 changeCount 
         // 变量，进行迭代判断。
         cudaerrcode = cudaMemcpy(&changeCount, devchangecount, sizeof (int),
                                  cudaMemcpyDeviceToHost);
         if (cudaerrcode != cudaSuccess) {
             // FAIL_THIN_IMAGE_FREE;
             return CUDA_ERROR;
         }

    }

    if (i % 2 == 0)
        cudaerrcode = cudaMemcpyPeer(outimg->imgData, outsubimgCud.deviceId, 
                                          tempimg->imgData, tempsubimgCud.deviceId, 
                                          tempsubimgCud.pitchBytes * tempimg->height);
     // 细化结束后释放申请的变量空间。
     cudaFree(devchangecount);
     ImageBasicOp::deleteImage(tempimg);
     return NO_ERROR;
}
