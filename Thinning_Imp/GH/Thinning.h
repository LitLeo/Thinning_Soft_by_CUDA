// Thinning.h
// 创建人：杨伟光
//
// 细化图像边界（Thinning）
// 功能说明：将灰度图像转化为 2 值图像，其内的 HEIGHT PIXEL 表示 EDGE，
//           实现将 EDGE 细化成 1 PIXEL 宽度的功能。且 HEIGHT PIXEL 和
//           LOW PIXEL 可由用户自己定义。


#ifndef __THINNING_H__
#define __THINNING_H__

#include "Image.h"
#include "ErrorCode.h"

// 类：Thinning（细化图像边界算法）
// 继承自：无。
// 实现了图像的细化算法。通过图像法和 PATTERN 表法对图像进行细化，实现将图像
// 细化成一个像素宽度的功能。
class Thinning {

protected:

    // 成员变量：highPixel（高像素）
    // 图像内高像素的像素值，可由用户定义。
    unsigned char highPixel;
    
    // 成员变量：lowPixel（低像素）
    // 图像内低像素的像素值，可由用户定义。
    unsigned char lowPixel;

    // 成员变量：imgCon（图像与坐标集之间的转化器）
    // 当参数为坐标集时，实现坐标集与图像的相互转化。
    // ImgConvert imgCon;
    
public:
    unsigned int DEF_BLOCK_X;
    unsigned int DEF_BLOCK_Y;

    // 构造函数：Thinning
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Thinning()
    {
        this->highPixel = 255;  // 高像素值默认为 255。
        this->lowPixel = 0;     // 低像素值默认为 0。 
        DEF_BLOCK_Y = 8;
        DEF_BLOCK_X = 32;
    }

    // 构造函数：Thinning
    // 有参数版本的构造函数，根据需要给定各个参数，这些参数值在程序运行过程中
    // 还是可以改变的。
    __host__ __device__
    Thinning(
            unsigned char highpixel,  // 高像素 
            unsigned char lowpixel    // 低像素
    ) {
        this->highPixel = 255;        // 高像素值默认为 255。
        this->lowPixel = 0;           // 低像素值默认为 0。

        // 根据参数列表中的值设定成员变量的初值。
        this->setHighLowPixel(highPixel, lowPixel);
    }

    // 成员函数：getHighPixel（获取高像素的值）
    // 获取成员变量 highPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 hignPixel 的值。
    getHighPixel() const
    { 
        // 返回 highPixel 成员变量的值。
        return highPixel;   
    }

    // 成员函数：setHighPixel（设置高像素）
    // 设置成员变量 highPixel 的值。
    __host__ __device__ int          // 返回值：若函数正确执行，返回 NO_ERROR。
    setHighPixel(                                                       
            unsigned char highpixel  // 高像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == this->lowPixel)
            return INVALID_DATA;

        // 将 highPixel 成员变量赋成新值
        this->highPixel = highpixel;

        return NO_ERROR;
    }

    // 成员函数：getLowPixel（获取低像素的值）
    // 获取成员变量 lowPixel 的值。
    __host__ __device__ unsigned char  // 返回值：返回 lowPixel 的值。
    getLowPixel() const
    { 
        // 返回 lowPixel 成员变量的值。
        return lowPixel;   
    }

    // 成员函数：setLowPixel（设置低像素）
    // 设置成员变量 lowPixel 的值。
    __host__ __device__ int         // 返回值：若函数正确执行，返回 NO_ERROR。
    setLowPixel(
            unsigned char lowpixel  // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (this->highPixel == lowpixel)
            return INVALID_DATA;

        // 将 lowPixel 成员变量赋成新值。
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

    // 成员函数：setHighLowPixel（设置高低像素）
    // 设置成员变量 highPixel 和 lowPixel 的值。
    __host__ __device__ int           // 返回值：函数正确执行，返回 NO_ERROR。
    setHighLowPixel(
            unsigned char highpixel,  // 高像素的像素值
            unsigned char lowpixel    // 低像素的像素值
    ) {
        // 如果高像素和低像素相等，则报错。
        if (highpixel == lowpixel)
            return INVALID_DATA;

        // 将 highPixel 成员变量赋成新值。
        this->highPixel = highpixel;

        // 将 lowPixel 成员变量赋成新值。
        this->lowPixel = lowpixel;

        return NO_ERROR;
    }

	__host__ int 
	thinGpu(
			Image *inimg,
			Image *outimg
	);

    __host__ int 
    thinGpuPt(
            Image *inimg,
            Image *outimg
    );

    __host__ int 
    thinGpuPPt(
            Image *inimg,
            Image *outimg
    );
    
};

#endif

