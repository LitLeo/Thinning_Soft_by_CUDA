// Thinning.h
// 创建人：杨伟光
// 实现 Zhang-sune 细化算法

#ifndef __THINNING_H__
#define __THINNING_H__

#include "Image.h"
#include "ErrorCode.h"

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

    // 构造函数：Thinning
    // 无参数版本的构造函数，所有的成员变量皆初始化为默认值。
    __host__ __device__
    Thinning()
    {
        this->highPixel = 255;  // 高像素值默认为 255。
        this->lowPixel = 0;     // 低像素值默认为 0。 
    }

    __host__ int thin(Image *inimg, Image *outimg);
    __host__ int thinFour(Image *inimg, Image *outimg);
    __host__ int thinPt(Image *inimg, Image *outimg);
    // __host__ int thinPtCon(Image *inimg, Image *outimg);
    __host__ int thinPtFour(Image *inimg, Image *outimg);
};

#endif

