#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define HIGH 255
#define LOW 0

// x4 x3 x2 y4 
// x5 x  x1 y5
// x6 x7 x8 y6
// y1 y2 y3

void PetThinSub1(Mat& tempimg, Mat& outimg, int& changecount)
{
    for (int r = 2; r < tempimg.rows - 2; r++) {
        for (int c = 2; c < tempimg.cols - 2; c++) {
            // if black
            if (tempimg.at<uchar>(r,c)) {
                uchar x1 = tempimg.at<uchar>(r,     c + 1);
                uchar x2 = tempimg.at<uchar>(r - 1, c + 1);
                uchar x3 = tempimg.at<uchar>(r - 1, c);
                uchar x4 = tempimg.at<uchar>(r - 1, c - 1);
                uchar x5 = tempimg.at<uchar>(r,     c - 1);
                uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                uchar x7 = tempimg.at<uchar>(r + 1, c);
                uchar x8 = tempimg.at<uchar>(r + 1, c + 1);
                // uchar y1 = tempimg.at<uchar>(r + 2, c - 1);
                uchar y2 = tempimg.at<uchar>(r + 2, c);
                // uchar y3 = tempimg.at<uchar>(r + 2, c + 1);
                // uchar y4 = tempimg.at<uchar>(r - 1, c + 2);
                uchar y5 = tempimg.at<uchar>(r,     c + 2);
                // uchar y6 = tempimg.at<uchar>(r + 1, c + 2);
                int A  = (x2 ^ x3) + (x3 ^ x4) + (x4 ^ x5) + (x5 ^ x6) + 
                         (x6 ^ x7) + (x7 ^ x8) + (x8 ^ x1) + (x1 ^ x2);
                int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
                int R = x1 && x7 && x8 &&
                       ((!y5 && x2 && x3 && !x5) || (!y2 && !x3 && x5 && x6));
                if (A == 2 && B >= 2 && B <= 6 && R == 0) {
                    outimg.at<uchar>(r,c) = LOW;
                    changecount ++; 
                }
            }   
        }
    }
}
void PetThinSub2(Mat& tempimg, Mat& outimg, int& changecount)
{
    for (int r = 2; r < tempimg.rows - 2; r++) {
        for (int c = 2; c < tempimg.cols - 2; c++) {
            // if black
            if (tempimg.at<uchar>(r,c)) {
                uchar x1 = tempimg.at<uchar>(r,     c + 1);
                uchar x2 = tempimg.at<uchar>(r - 1, c + 1);
                uchar x3 = tempimg.at<uchar>(r - 1, c);
                uchar x4 = tempimg.at<uchar>(r - 1, c - 1);
                uchar x5 = tempimg.at<uchar>(r,     c - 1);
                uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                uchar x7 = tempimg.at<uchar>(r + 1, c);
                uchar x8 = tempimg.at<uchar>(r + 1, c + 1);
                // uchar y1 = tempimg.at<uchar>(r + 2, c - 1);
                // uchar y2 = tempimg.at<uchar>(r + 2, c);
                // uchar y3 = tempimg.at<uchar>(r + 2, c + 1);
                // uchar y4 = tempimg.at<uchar>(r - 1, c + 2);
                // uchar y5 = tempimg.at<uchar>(r,     c + 2);
                // uchar y6 = tempimg.at<uchar>(r + 1, c + 2);
                int S0 = (x3&&x7) || (x5&&x1);
                int S1 = (x1 && !x6 && (!x4 || x3)) || (x3 && !x8 && (!x6 || x5)) ||
                    (x7 && !x4 && (!x2 || x1)) || (x5 && !x2 && (!x8 || x7));
                int B  = x2 + x3 + x4 + x5 + x6 + x7 + x8 + x1;
                int R = (x3 && (x1&&!x8 || x5&&!x6)) || (x7 && (!x5&&!x8 || !x1&&!x6));
                if ((!S0 && S1) && R == 0 && B >= 3) {
                    outimg.at<uchar>(r,c) = LOW;
                    changecount ++; 
                }
            }   
        }
    }
}

// 细化函数
int PetThin(Mat& inimg, Mat& outimg)
{
    Mat tempimg;
    inimg.copyTo(outimg);
    // inimg.copyTo(tempimg);
    
    outimg /= HIGH;
    // printMat(outimg);
    
    int changecount = 1;

    while (changecount != 0)
    {
        outimg.copyTo(tempimg);
        // printMat(outimg);
        changecount = 0;

        PetThinSub1(tempimg, outimg, changecount);
        //printMat(outimg);
        outimg.copyTo(tempimg);
        PetThinSub2(tempimg, outimg, changecount);
        //printMat(outimg);
    }
    outimg *= HIGH;
    return 0;
}

int main(int argc, const char **argv)
{
    if(argc < 2)
    {
      cout << "Please input image!" << endl;
      return 0;
    } 
    // Mat inimg = imread("32-32.bmp", CV_8UC1);
    Mat inimg = imread(argv[1], CV_8UC1);
    Mat outimg(inimg.size(), CV_8UC1);
    
    clock_t start,finish;
    double totaltime;

    start=clock();
    for (int i = 0; i < 100; ++i)
        PetThin(inimg, outimg);
    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\nPetThin runTime = "<<totaltime <<"ms！"<<endl;

    string str(argv[1]);
    str = "out-" + str.substr(10);
    imwrite( str, outimg);
    
    cv::waitKey(0);

    return 0;
}