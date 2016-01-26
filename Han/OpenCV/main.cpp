#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

#define HIGH 255
#define LOW 0
#define MoreThanOne(x1, x2, x3, x4, x5, x6, x7, x8, qt) ( (x1) >= qt || (x2) >= qt || (x3) >= qt || (x4) >= qt || (x5) >= qt || (x6) >= qt || (x7) >= qt || (x8) >= qt ? 1 : 0)

#define notZero2(x1, x2) ( (x1) != 0 && (x2) != 0 ? 1 : 0)
#define notZero3(x1, x2, x3) ( (x1) != 0 && (x2) != 0 && (x3) != 0 ? 1 : 0)
#define notZero4(x1, x2, x3, x4) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 ? 1 : 0)
#define notZero5(x1, x2, x3, x4, x5) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 ? 1 : 0)
#define notZero6(x1, x2, x3, x4, x5, x6) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 && (x6) != 0 ? 1 : 0)
#define notZero7(x1, x2, x3, x4, x5, x6, x7) ( (x1) != 0 && (x2) != 0 && (x3) != 0 && (x4) != 0 && (x5) != 0 && (x6) != 0 && (x7) != 0 ? 1 : 0)


void printMat(Mat im, int d = 255)
{
    im /= d;
    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            cout << (int)im.at<uchar>(i,j) << " ";
        }
        cout << endl;
    }

    cout << endl << endl << endl;
    im *= d;
}

// 计算八邻域高像素点的数量
int calWeight(Mat& inimg, Mat& weight)
{
    for (int r = 1; r < inimg.rows-1; r++)
        for (int c = 1; c < inimg.cols-1; c++)
        {
            if(inimg.at<uchar>(r,c) == HIGH) {
                uchar x1 = inimg.at<uchar>(r - 1, c-1);
                uchar x2 = inimg.at<uchar>(r - 1, c);
                uchar x3 = inimg.at<uchar>(r - 1, c+1);
                uchar x4 = inimg.at<uchar>(r, c-1);
                uchar x5 = inimg.at<uchar>(r, c+1);
                uchar x6 = inimg.at<uchar>(r + 1, c-1);
                uchar x7 = inimg.at<uchar>(r + 1, c);
                uchar x8 = inimg.at<uchar>(r + 1, c+1);

                weight.at<uchar>(r,c) = (x1 == HIGH) + (x2 == HIGH) + (x3 == HIGH) + (x4 == HIGH) + 
                                        (x5 == HIGH) + (x6 == HIGH) + (x7 == HIGH) + (x8 == HIGH);
            }
        }
    
    return 0;
}
// x1 x2 x3
// x8    x4
// x7 x6 x5
int HanThinning(Mat& inimg, Mat& outimg)
{
    Mat weight(inimg.size(), CV_8SC1);
    
    inimg.copyTo(outimg);
    
    
    uchar flag = 1;
    // 迭代
    while (flag != 0){
        flag = 0;
        weight = 0; // 初始化
        calWeight(outimg, weight);
       //  printMat(weight, 1);
        for (int r = 1; r < inimg.rows-1; r++)
            for (int c = 1; c < inimg.cols-1; c++) {
                // 忽略低像素点情况和dot情况
                if (outimg.at<uchar>(r,c) == HIGH && (weight.at<uchar>(r,c) > 0 && weight.at<uchar>(r,c) < 8)) {
                    uchar x1 = weight.at<uchar>(r - 1, c-1);
                    uchar x2 = weight.at<uchar>(r - 1, c);
                    uchar x3 = weight.at<uchar>(r - 1, c+1);
                    uchar x4 = weight.at<uchar>(r, c+1);
                    uchar x5 = weight.at<uchar>(r + 1, c+1);
                    uchar x6 = weight.at<uchar>(r + 1, c);
                    uchar x7 = weight.at<uchar>(r + 1, c-1);
                    uchar x8 = weight.at<uchar>(r, c-1);
                
                    // 判断每一个点的权重值
                    switch (weight.at<uchar>(r,c))
                    {
                    case 1 :
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                        {
                            outimg.at<uchar>(r,c) = LOW;
                            flag = 1;
                        }
                        break;
                    case 2 : 
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                            if (x1!=0&&x2!=0 || x2!=0&&x3!=0 || x3!=0&&x4!=0 || x4!=0&&x5!=0 || x5!=0&&x6!=0 || x6!=0&&x7!=0 ||
                                x7!=0&&x8!=0 || x8!=0&&x1!=0 || x2!=0&&x4!=0 || x4!=0&&x6!=0 || x6!=0&&x8!=0 || x8!=0&&x2!=0)
                            {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 3 : 
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,7))
                            if (notZero3(x6,x7,x8) || notZero3(x1,x2,x3) || notZero3(x1,x7,x8) || notZero3(x6,x7,x5) || 
                                notZero3(x3,x4,x5) || notZero3(x2,x3,x4) || notZero3(x4,x5,x6) || notZero3(x8,x1,x2) ||
                                notZero3(x6,x7,x4) || notZero3(x6,x1,x8) || notZero3(x6,x3,x4) || notZero3(x6,x5,x8))
                            {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 4 : 
                        if (notZero4(x1,x2,x3,x4) || notZero4(x1,x2,x7,x8) || notZero4(x1,x2,x3,x8) || notZero4(x1,x6,x7,x8) ||
                            notZero4(x5,x6,x7,x8) || notZero4(x4,x5,x6,x7) || notZero4(x3,x4,x5,x6) || notZero4(x5,x2,x3,x4) ||
                            notZero4(x6,x7,x3,x4) || notZero4(x1,x8,x5,x6))
                        {
                            outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                        }
                        break;
                    case 5 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (notZero5(x7,x8,x1,x2,x3) || notZero5(x7,x8,x1,x5,x6) || notZero5(x3,x4,x5,x6,x7) || notZero5(x1,x2,x3,x4,x5) ||
                                notZero5(x4,x5,x6,x7,x8) || notZero5(x6,x7,x8,x1,x2) || notZero5(x1,x2,x3,x4,x8) || notZero5(x2,x3,x4,x5,x6)) {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 6 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (notZero6(x3,x4,x5,x6,x7,x8) || notZero6(x1,x2,x3,x6,x7,x8) || notZero6(x1,x2,x5,x6,x7,x8) || notZero6(x1,x4,x5,x6,x7,x8) ||
                                notZero6(x1,x2,x3,x4,x7,x8) || notZero6(x3,x4,x5,x6,x7,x2) || notZero6(x3,x4,x5,x6,x1,x2) || notZero6(x1,x2,x3,x4,x5,x8)) {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 7 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (notZero7(x1,x2,x3,x5,x6,x7,x8) || notZero7(x1,x3,x4,x5,x6,x7,x8) ||
                                notZero7(x1,x2,x3,x4,x5,x6,x7) || notZero7(x1,x2,x3,x4,x5,x7,x8)){
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    default:
                        break;
                    }
                }
            }
    }

    return 0;
}

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


int HanThinningPt(Mat& inimg, Mat& outimg)
{
    Mat weight(inimg.size(), CV_8SC1);
    inimg.copyTo(outimg)
    uchar flag = 1;
    // 迭代
    while (flag != 0){
        flag = 0;
        weight = 0; // 初始化
        calWeight(outimg, weight);
        // printMat(weight, 1);
        for (int r = 1; r < inimg.rows-1; r++)
            for (int c = 1; c < inimg.cols-1; c++) {
                // 忽略低像素点情况和dot情况
                if (outimg.at<uchar>(r,c) == HIGH && (weight.at<uchar>(r,c) > 0 && weight.at<uchar>(r,c) < 8)) {
                    uchar x1 = weight.at<uchar>(r - 1, c-1);
                    uchar x2 = weight.at<uchar>(r - 1, c);
                    uchar x3 = weight.at<uchar>(r - 1, c+1);
                    uchar x4 = weight.at<uchar>(r, c+1);
                    uchar x5 = weight.at<uchar>(r + 1, c+1);
                    uchar x6 = weight.at<uchar>(r + 1, c);
                    uchar x7 = weight.at<uchar>(r + 1, c-1);
                    uchar x8 = weight.at<uchar>(r, c-1);
                    uchar index = (x1!=0) * 1 + (x2!=0) * 2 + (x3!=0) * 4 + (x4!=0) * 8 + 
                                  (x5!=0) * 16 + (x6!=0) * 32 + (x7!=0) * 64 + (x8!=0) * 128; 

                    // 判断每一个点的权重值
                    switch (weight.at<uchar>(r,c))
                    {
                    case 1 :
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                        {
                            outimg.at<uchar>(r,c) = LOW;
                            flag = 1;
                        }
                        break;
                    case 2 : 
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,3))
                            if (lut[index])
                            {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 3 : 
                        if (MoreThanOne(x1,x2,x3,x4,x5,x6,x7,x8,7))
                            if (lut[index + 256])
                            {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 4 : 
                        if (lut[index + 512])
                        {
                            outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                        }
                        break;
                    case 5 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (lut[index + 768]) {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 6 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (lut[index + 1024]) {
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    case 7 : 
                        if (x1==8 || x2==8 || x3==8 || x4==8 || x5==8 || x6==8 || x7==8 || x8==8)
                            if (lut[index + 1280]){
                                outimg.at<uchar>(r,c) = LOW;
                                flag = 1;
                            }
                        break;
                    default:
                        break;
                    }
                }
            }
    }

    return 0;
}


int main(int argc, char const **argv)
{ 
    if(argc < 2)
    {
      cout << "Please input image!" << endl;
      return 0;
    }
    // Mat inimg = imread("256-256.bmp", CV_8UC1);
    Mat inimg = imread(argv[1], CV_8UC1);
    Mat outimg(inimg.size(), CV_8UC1);
    
    clock_t start,finish;
    double totaltime;

    // start=clock();
    // for (int i = 0; i < 100; ++i)
    //     HanThinning(inimg, outimg);
    // finish=clock();
    // totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    // cout<<"\nHanThinning runTime = "<<totaltime <<"ms！"<<endl;
    
    start=clock();
    for (int i = 0; i < 100; ++i)
        HanThinningPt(inimg, outimg);
    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\nHanThinningPt runTime = "<<totaltime <<"ms！"<<endl;

    string str(argv[1]);
    str = "out-" + str.substr(10);
    imwrite( str, outimg);
    waitKey();
    return 0;
}
