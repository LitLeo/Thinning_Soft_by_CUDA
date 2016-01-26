// Thinning.cu
// 实现二值图像的细化算法

#include "Thinning.h"
#include <iostream>
#include <stdio.h>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块尺寸。
#define DEF_BLOCK_X  32
#define DEF_BLOCK_Y   8

#define uchar unsigned char

// 宏：DEF_PATTERN_SIZE
// 定义了 PATTERN 表的默认大小。
#define DEF_PATTERN_SIZE  512

#define HIGH 255
#define LOW 0
#define MoreThanOne(x1, x2, x3, x4, x5, x6, x7, x8, qt) ( (x1) >= qt || (x2) >= qt || (x3) >= qt || (x4) >= qt || (x5) >= qt || (x6) >= qt || (x7) >= qt || (x8) >= qt ? 1 : 0)

#define notZero2(x1, x2) ( (x1) != 0 && (x2) != 0 ? 1 : 0)
#define notZero3(x1, x2, x3) ( (x1) != 0 && (x2) != 0 && (x3) != 0 ? 1 : 0)
#define notZero4(x1, x2, x3, x4) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 ? 1 : 0)
#define notZero5(x1, x2, x3, x4, x5) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 ? 1 : 0)
#define notZero6(x1, x2, x3, x4, x5, x6) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 && (x6) != 0 ? 1 : 0)
#define notZero7(x1, x2, x3, x4, x5, x6, x7) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 && (x6) != 0 && (x7) != 0 ? 1 : 0)

static __global__ void _calWightKer(ImageCuda inimg, ImageCuda weight)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= inimg.imgMeta.width - 1 || 
         r >= inimg.imgMeta.height - 1 || c < 1 || r < 1)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * inimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = inimg.imgMeta.imgData + curpos ;

    if (*outptr == HIGH) {
        unsigned char x1 = inimg.imgMeta.imgData[curpos - inimg.pitchBytes - 1];
        unsigned char x2 = inimg.imgMeta.imgData[curpos - inimg.pitchBytes];
        unsigned char x3 = inimg.imgMeta.imgData[curpos - inimg.pitchBytes + 1];
        unsigned char x4 = inimg.imgMeta.imgData[curpos - 1];
        unsigned char x5 = inimg.imgMeta.imgData[curpos + 1];
        unsigned char x6 = inimg.imgMeta.imgData[curpos + inimg.pitchBytes - 1];
        unsigned char x7 = inimg.imgMeta.imgData[curpos + inimg.pitchBytes];
        unsigned char x8 = inimg.imgMeta.imgData[curpos + inimg.pitchBytes + 1];

        weight.imgMeta.imgData[curpos] = (x1 == HIGH) + (x2 == HIGH) + (x3 == HIGH) + (x4 == HIGH) + 
                                         (x5 == HIGH) + (x6 == HIGH) + (x7 == HIGH) + (x8 == HIGH);
    }
}

static __global__ void _thinHanKer(ImageCuda weight, ImageCuda outimg, int *devchangecount)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 1 || 
         r >= outimg.imgMeta.height - 1 || c < 1 || r < 1)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    

    // 忽略低像素点情况和dot情况
    if (*outptr == HIGH && (weight.imgMeta.imgData[curpos] > 0 && weight.imgMeta.imgData[curpos] < 8)) {
        uchar x1 = weight.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        uchar x2 = weight.imgMeta.imgData[curpos - outimg.pitchBytes];
        uchar x3 = weight.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        uchar x4 = weight.imgMeta.imgData[curpos + 1];
        uchar x5 = weight.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        uchar x6 = weight.imgMeta.imgData[curpos + outimg.pitchBytes];
        uchar x7 = weight.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        uchar x8 = weight.imgMeta.imgData[curpos - 1];

        // 判断每一个点的权重值
        switch (weight.imgMeta.imgData[curpos])
        {
        case 1 :
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
            {
                *outptr = LOW;
                *devchangecount = 1;
            }
            break;
        case 2 : 
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                if (x1!=0&&x2!=0 || x2!=0&&x3!=0 || x3!=0&&x4!=0 || x4!=0&&x5!=0 || x5!=0&&x6!=0 || x6!=0&&x7!=0 ||
                    x7!=0&&x8!=0 || x8!=0&&x1!=0 || x2!=0&&x4!=0 || x4!=0&&x6!=0 || x6!=0&&x8!=0 || x8!=0&&x2!=0)
                {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 3 : 
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,7))
                if (notZero3(x6,x7,x8) || notZero3(x1,x2,x3) || notZero3(x1,x7,x8) || notZero3(x6,x7,x5) || 
                    notZero3(x3,x4,x5) || notZero3(x2,x3,x4) || notZero3(x4,x5,x6) || notZero3(x8,x1,x2) ||
                    notZero3(x6,x7,x4) || notZero3(x6,x1,x8) || notZero3(x6,x3,x4) || notZero3(x6,x5,x8))
                {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 4 : 
            if (notZero4(x1,x2,x3,x4) || notZero4(x1,x2,x7,x8) || notZero4(x1,x2,x3,x8) || notZero4(x1,x6,x7,x8) ||
                notZero4(x5,x6,x7,x8) || notZero4(x4,x5,x6,x7) || notZero4(x3,x4,x5,x6) || notZero4(x5,x2,x3,x4) ||
                notZero4(x6,x7,x3,x4) || notZero4(x1,x8,x5,x6))
            {
                *outptr = LOW;
                    *devchangecount = 1;
            }
            break;
        case 5 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (notZero5(x7,x8,x1,x2,x3) || notZero5(x7,x8,x1,x5,x6) || notZero5(x3,x4,x5,x6,x7) || notZero5(x1,x2,x3,x4,x5) ||
                    notZero5(x4,x5,x6,x7,x8) || notZero5(x6,x7,x8,x1,x2) || notZero5(x1,x2,x3,x4,x8) || notZero5(x2,x3,x4,x5,x6)) {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 6 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (notZero6(x3,x4,x5,x6,x7,x8) || notZero6(x1,x2,x3,x6,x7,x8) || notZero6(x1,x2,x5,x6,x7,x8) || notZero6(x1,x4,x5,x6,x7,x8) ||
                    notZero6(x1,x2,x3,x4,x7,x8) || notZero6(x3,x4,x5,x6,x7,x2) || notZero6(x3,x4,x5,x6,x1,x2) || notZero6(x1,x2,x3,x4,x5,x8)) {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 7 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (notZero7(x1,x2,x3,x5,x6,x7,x8) || notZero7(x1,x3,x4,x5,x6,x7,x8) ||
                    notZero7(x1,x2,x3,x4,x5,x6,x7) || notZero7(x1,x2,x3,x4,x5,x7,x8)){
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        default:
            break;
        }
    }
}

// 直接并行化
// 线程数，处理多少个点有多少线程数
__host__ int Thinning::thinHan(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    // 声明所有中间变量并初始化为空。
    Image *weight = NULL;
    int *devchangecount = NULL;

    // 记录细化点数的变量，位于 host 端。
    int changeCount;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&weight);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(weight, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
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
    ImageCuda weightsubimgCud;
    errcode = ImageBasicOp::roiSubImage(weight, &weightsubimgCud);
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

    /*gridsize.x = 1;*/
    /*gridsize.y = 1;//(outsubimgCud.imgMeta.height + blocksize.y - 1) / blocksize.y;*/
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
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 初始化 weight 为 0
        cudaerrcode = cudaMemset (weight->imgData, 0, sizeof(unsigned char) * weight->width * weight->height);
        if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
        }
        
        _calWightKer<<<gridsize, blocksize>>>(outsubimgCud, weightsubimgCud);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
            return CUDA_ERROR;
        }

        // 调用核函数，开始第一步细化操作。
        _thinHanKer<<<gridsize, blocksize>>>(weightsubimgCud, outsubimgCud, devchangecount);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
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
    ImageBasicOp::deleteImage(weight);

    return NO_ERROR;
}

// GPU版本2，优化分支，使用Pattern表法，Pattern表位于global内存中
static __global__ void _thinHanPtKer(ImageCuda weight, ImageCuda outimg, 
                                    int *devchangecount, unsigned char *dev_lut)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 1 || 
         r >= outimg.imgMeta.height - 1 || c < 1 || r < 1)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    

    // 忽略低像素点情况和dot情况
    if (*outptr == HIGH && (weight.imgMeta.imgData[curpos] > 0 && weight.imgMeta.imgData[curpos] < 8)) {
        uchar x1 = weight.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        uchar x2 = weight.imgMeta.imgData[curpos - outimg.pitchBytes];
        uchar x3 = weight.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        uchar x4 = weight.imgMeta.imgData[curpos + 1];
        uchar x5 = weight.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        uchar x6 = weight.imgMeta.imgData[curpos + outimg.pitchBytes];
        uchar x7 = weight.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        uchar x8 = weight.imgMeta.imgData[curpos - 1];
        uchar index = (x1!=0) * 1 + (x2!=0) * 2 + (x3!=0) * 4 + (x4!=0) * 8 + 
                      (x5!=0) * 16 + (x6!=0) * 32 + (x7!=0) * 64 + (x8!=0) * 128; 

        // 判断每一个点的权重值
        switch (weight.imgMeta.imgData[curpos])
        {
        case 1 :
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
            {
                *outptr = LOW;
                *devchangecount = 1;
            }
            break;
        case 2 : 
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                if (dev_lut[index])
                {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 3 : 
            if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,7))
                if (dev_lut[index + 256])
                {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 4 : 
            if (dev_lut[index + 512])
            {
                *outptr = LOW;
                    *devchangecount = 1;
            }
            break;
        case 5 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (dev_lut[index + 768]) {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 6 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (dev_lut[index + 1024]) {
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        case 7 : 
            if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                if (dev_lut[index + 1280]){
                    *outptr = LOW;
                    *devchangecount = 1;
                }
            break;
        default:
            break;
        }
    }
}

__host__ int Thinning::thinHanPt(Image *inimg, Image *outimg)
{
    // 局部变量，错误码。
    int errcode;  
    cudaError_t cudaerrcode; 

    // 检查输入图像，输出图像是否为空。
    if (inimg == NULL || outimg == NULL)
        return NULL_POINTER;

    unsigned char *dev_lut;
    unsigned char lut[1536] = 
    {
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 

        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1
    };
    cudaerrcode = cudaMalloc((void **)&dev_lut, sizeof (unsigned char) * 1536);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;

    cudaerrcode = cudaMemcpy(dev_lut, lut, sizeof(unsigned char) * 1536, cudaMemcpyHostToDevice);
    if (cudaerrcode != cudaSuccess) 
        return CUDA_ERROR;


    // 声明所有中间变量并初始化为空。
    Image *weight = NULL;
    int *devchangecount = NULL;

    // 记录细化点数的变量，位于 host 端。
    int changeCount;

    // 记录细化点数的变量，位于 device 端。并为其申请空间。
    cudaerrcode = cudaMalloc((void **)&devchangecount, sizeof (int));
    if (cudaerrcode != cudaSuccess) {
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&weight);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(weight, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
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
    ImageCuda weightsubimgCud;
    errcode = ImageBasicOp::roiSubImage(weight, &weightsubimgCud);
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
            // FAIL_THIN_IMAGE_FREE;
            return CUDA_ERROR;
        }

        // 初始化 weight 为 0
        cudaerrcode = cudaMemset (weight->imgData, 0, sizeof(unsigned char) * weight->width * weight->height);
        if (cudaerrcode != cudaSuccess) {
         return CUDA_ERROR;
        }
        
        _calWightKer<<<gridsize, blocksize>>>(outsubimgCud, weightsubimgCud);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
            return CUDA_ERROR;
        }

        // 调用核函数，开始第一步细化操作。
        _thinHanPtKer<<<gridsize, blocksize>>>(weightsubimgCud, outsubimgCud, devchangecount, dev_lut);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
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
    ImageBasicOp::deleteImage(weight);

    return NO_ERROR;
}
/*
// GPU版本3，优化读取，将Pattern存到常量内存中
__constant__ unsigned char con_lutthin[256] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 
    1, 1, 0, 0, 1, 1, 0, 0 };

static __global__ void _thinHanPtConKer(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
     // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                              isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (con_lutthin[index] == 1) {
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
        }       
    }
}

__host__ int Thinning::thinHanPtCon(Image *inimg, Image *outimg)
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
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
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
            
        // 调用核函数，开始第一步细化操作。
        _thinHanPtConKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
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

// GPU版本5，GPU版本1的基础上使用单线程处理四个点
static __global__ void _thinHanFourKer(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
    // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }
        
        if (isHigh4(x1, x4, x6, x7) && isLow2(x3, x5) || // 1
            isHigh4(x1, x2, x4, x6) && isLow2(x5, x8) || // 2
            isHigh4(x1, x2, x3, x5) && isLow2(x6, x7) || // 3
            isHigh4(x1, x2, x3, x4) && isLow2(x7, x8) || // 4
            isLow4(x3, x5, x7, x8) && isHigh2(x1, x4) || // 5
            isLow4(x5, x6, x7, x8) && isHigh2(x1, x2) || // 6
            isLow1(x5) && isHigh7(x1, x2, x3, x4, x6, x7, x8) || // 7
            isLow1(x7) && isHigh7(x1, x2, x3, x4, x5, x6, x8) || // 8
            isLow4(x2, x3, x5, x8) && isHigh2(x4, x6) || // 9
            isLow4(x1, x2, x3, x5) && isHigh2(x6, x7) || // 10
            isLow4(x4, x6, x7, x8) && isHigh2(x2, x3) || // 11
            isLow4(x1, x4, x6, x7) && isHigh2(x3, x5) || // 12
            isLow4(x1, x2, x3, x4) && isHigh2(x7, x8) || // 13
            isLow4(x1, x2, x4, x6) && isHigh2(x5, x8) || // 14
            isLow1(x4) && isHigh7(x1, x2, x3, x5, x6, x7, x8) || // 15
            isLow1(x2) && isHigh7(x1, x3, x4, x5, x6, x7, x8) || // 16
            isHigh4(x3, x5, x7, x8) && isLow2(x1, x4) || // 17
            isHigh4(x2, x3, x5, x8) && isLow2(x4, x6) || // 18
            isHigh4(x5, x6, x7, x8) && isLow2(x1, x2) || // 19
            isHigh4(x4, x6, x7, x8) && isLow2(x2, x3) // 20
            ) {
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
        }       
    }

    for (int i = 0; i < 3; ++i) {
        if (++r >= outimg.imgMeta.height - 2)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;

        // 如果目标像素点的像素值为低像素, 则不进行细化处理。
        if (isHigh1(*outptr)) {
            x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
            x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
            x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
            x4 = tempimg.imgMeta.imgData[curpos - 1];
            x5 = tempimg.imgMeta.imgData[curpos + 1];
            x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
            x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
            x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
            unsigned char x9,x10,x11;
            if(isHigh1(x7)) {
                x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
                x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

                if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                    isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                    isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                        return ;
            } 
            if(isHigh1(x2)) {
                // w is down
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
                x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
                x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

                if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
                    return ;
                } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                           isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                    return ;
            }
            if(isHigh1(x5)) {
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
                x10 = tempimg.imgMeta.imgData[curpos + 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

                if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                    isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                    isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                    return ;
            }
            if(isHigh1(x4)){
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
                x10 = tempimg.imgMeta.imgData[curpos - 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
                if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
                    return ;
                } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                           isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                    return ;
            }
            
            if (isHigh4(x1, x4, x6, x7) && isLow2(x3, x5) || // 1
                isHigh4(x1, x2, x4, x6) && isLow2(x5, x8) || // 2
                isHigh4(x1, x2, x3, x5) && isLow2(x6, x7) || // 3
                isHigh4(x1, x2, x3, x4) && isLow2(x7, x8) || // 4
                isLow4(x3, x5, x7, x8) && isHigh2(x1, x4) || // 5
                isLow4(x5, x6, x7, x8) && isHigh2(x1, x2) || // 6
                isLow1(x5) && isHigh7(x1, x2, x3, x4, x6, x7, x8) || // 7
                isLow1(x7) && isHigh7(x1, x2, x3, x4, x5, x6, x8) || // 8
                isLow4(x2, x3, x5, x8) && isHigh2(x4, x6) || // 9
                isLow4(x1, x2, x3, x5) && isHigh2(x6, x7) || // 10
                isLow4(x4, x6, x7, x8) && isHigh2(x2, x3) || // 11
                isLow4(x1, x4, x6, x7) && isHigh2(x3, x5) || // 12
                isLow4(x1, x2, x3, x4) && isHigh2(x7, x8) || // 13
                isLow4(x1, x2, x4, x6) && isHigh2(x5, x8) || // 14
                isLow1(x4) && isHigh7(x1, x2, x3, x5, x6, x7, x8) || // 15
                isLow1(x2) && isHigh7(x1, x3, x4, x5, x6, x7, x8) || // 16
                isHigh4(x3, x5, x7, x8) && isLow2(x1, x4) || // 17
                isHigh4(x2, x3, x5, x8) && isLow2(x4, x6) || // 18
                isHigh4(x5, x6, x7, x8) && isLow2(x1, x2) || // 19
                isHigh4(x4, x6, x7, x8) && isLow2(x2, x3) // 20
                ) {
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
            }       
        }

    }
}

__host__ int Thinning::thinHanFour(Image *inimg, Image *outimg)
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
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
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
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / blocksize.y * 4;

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
            
        // 调用核函数，开始第一步细化操作。
        _thinHanFourKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
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

// GPU版本4，使用Pattern表法后，单线程内逻辑大大降低。使单线程处理四个点，增大线程复杂度，减少线程数量
static __global__ void _thinHanPtConFourKer(ImageCuda tempimg, ImageCuda outimg, int *devchangecount)
{
     // c 和 r 分别表示线程处理的像素点的坐标的 x 和 y 分量 （其中，c 表示
    // column，r 表示 row ）。
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查第一个像素点是否越界，如果越界，则不进行处理，一方面节省计算资源，
    // 另一方面防止由于段错误导致程序崩溃。
    // 两边各有两个点不处理。
    if (c >= outimg.imgMeta.width - 2 || 
         r >= outimg.imgMeta.height - 2 || c < 2 || r < 2)
        return;

    // 定义目标点位置的指针。
    unsigned char *outptr;

    // 获取当前像素点在图像中的相对位置。
    // 从左上角第二行第二列开始计算。
    int curpos = (r) * outimg.pitchBytes + c ;

    // 获取当前像素点在图像中的绝对位置。
    outptr = outimg.imgMeta.imgData + curpos ;
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8;

    // 如果目标像素点的像素值为低像素, 则不进行细化处理。
    if (isHigh1(*outptr)) {
        x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
        x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
        x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
        x4 = tempimg.imgMeta.imgData[curpos - 1];
        x5 = tempimg.imgMeta.imgData[curpos + 1];
        x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
        x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
        x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
        unsigned char x9,x10,x11;
        if(isHigh1(x7)) {
            x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

            if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                    return ;
        } 
        if(isHigh1(x2)) {
            // w is down
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
            x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
            x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

            if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                       isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                return ;
        }
        if(isHigh1(x5)) {
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
            x10 = tempimg.imgMeta.imgData[curpos + 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

            if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                return ;
        }
        if(isHigh1(x4)){
            x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
            x10 = tempimg.imgMeta.imgData[curpos - 2];
            x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
            if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
                return ;
            } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                       isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                return ;
        }

        // 1   2   4
        // 8       16
        // 32  64  128
        unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                              isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
        
        if (con_lutthin[index] == 1) {
                outimg.imgMeta.imgData[curpos] = LOW;
                *devchangecount = 1;
        }       
    }

    for (int i = 0; i < 3; ++i) {
        if (++r >= outimg.imgMeta.height - 2)
            return ;
        curpos += outimg.pitchBytes;  

        // 获取当前像素点在图像中的绝对位置。
        outptr = outimg.imgMeta.imgData + curpos;

        // 如果目标像素点的像素值为低像素, 则不进行细化处理。
        if (isHigh1(*outptr)) {
            x1 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 1];
            x2 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes];
            x3 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 1];
            x4 = tempimg.imgMeta.imgData[curpos - 1];
            x5 = tempimg.imgMeta.imgData[curpos + 1];
            x6 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 1];
            x7 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes];
            x8 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 1];
            unsigned char x9,x10,x11;
            if(isHigh1(x7)) {
                x9 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 - 1];
                x10 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes * 2 + 1];

                if (isHigh5(x4,x5,x6,x7,x8) && isLow2(x2,x10) ||
                    isHigh3(x6,x7,x9) && isLow8(x1,x2,x3,x4,x5,x8,x10,x11) ||
                    isHigh3(x7,x8,x11) && isLow8(x1,x2,x3,x4,x5,x6,x9,x10))
                        return ;
            } 
            if(isHigh1(x2)) {
                // w is down
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 - 1];
                x10 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2];
                x11 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes * 2 + 1];

                if (isHigh5(x1,x2,x3,x4,x5) && isLow2(x7,x10)){
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
                    return ;
                } else if (isHigh3(x1,x2,x9) && isLow8(x3,x4,x5,x6,x7,x8,x10,x11) ||
                           isHigh3(x2,x3,x11) && isLow8(x1,x4,x5,x6,x7,x8,x9,x10)) 
                    return ;
            }
            if(isHigh1(x5)) {
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes + 2];
                x10 = tempimg.imgMeta.imgData[curpos + 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes + 2];

                if (isHigh5(x2,x3,x5,x7,x8) && isLow2(x4,x10) ||
                    isHigh3(x5,x8,x11) && isLow8(x1,x2,x3,x4,x6,x7,x9,x10) ||
                    isHigh3(x3,x5,x9) && isLow8(x1,x2,x4,x6,x7,x8,x10,x11))
                    return ;
            }
            if(isHigh1(x4)){
                x9 = tempimg.imgMeta.imgData[curpos - outimg.pitchBytes - 2];
                x10 = tempimg.imgMeta.imgData[curpos - 2];
                x11 = tempimg.imgMeta.imgData[curpos + outimg.pitchBytes - 2];
                if (isHigh5(x1,x2,x4,x6,x7) && isLow2(x5,x10)){
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
                    return ;
                } else if (isHigh3(x4,x6,x11) && isLow8(x1,x2,x3,x5,x7,x8,x9,x10) ||
                           isHigh3(x1,x4,x9) && isLow8(x2,x3,x5,x6,x7,x8,x10,x11)) 
                    return ;
            }

            // 1   2   4
            // 8       16
            // 32  64  128
            unsigned char index = isHigh1(x1) * 1 + isHigh1(x2) * 2 + isHigh1(x3) * 4 + isHigh1(x4) * 8 +
                                  isHigh1(x5) * 16 + isHigh1(x6) * 32 + isHigh1(x7) * 64 + isHigh1(x8) * 128;
            
            if (con_lutthin[index] == 1) {
                    outimg.imgMeta.imgData[curpos] = LOW;
                    *devchangecount = 1;
            }       
        }
    }
}

__host__ int Thinning::thinHanPtConFour(Image *inimg, Image *outimg)
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
        // FAIL_THIN_IMAGE_FREE;
        return CUDA_ERROR;
    }

    // 生成暂存图像。
    errcode = ImageBasicOp::newImage(&tempimg);
    if (errcode != NO_ERROR)
        return errcode;
    errcode = ImageBasicOp::makeAtCurrentDevice(tempimg, inimg->width, 
                                                inimg->height);
    if (errcode != NO_ERROR) {
        // FAIL_THIN_IMAGE_FREE;
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
    gridsize.y = (outsubimgCud.imgMeta.height + blocksize.y * 4 - 1) / blocksize.y * 4;

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
            
        // 调用核函数，开始第一步细化操作。
        _thinHanPtConFourKer<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount);
        if (cudaGetLastError() != cudaSuccess) {
            // 核函数出错，结束迭代函数，释放申请的变量空间。
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
*/