/**
 * Code for thinning a binary image using Guo-Hall algorithm.
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>

 using namespace std;
 using namespace cv;

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningGuoHallIteration(cv::Mat& im, int iter, int& diff)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

    for (int i = 1; i < im.rows - 1; i++)
    {
        for (int j = 1; j < im.cols - 1; j++)
        {
            if(im.at<uchar>(i,j) == 1)
            {
                uchar p1 = im.at<uchar>(i-1, j-1);
                uchar p2 = im.at<uchar>(i-1, j);
                uchar p3 = im.at<uchar>(i-1, j+1);
                uchar p4 = im.at<uchar>(i, j+1);
                uchar p5 = im.at<uchar>(i+1, j+1);
                uchar p6 = im.at<uchar>(i+1, j);
                uchar p7 = im.at<uchar>(i+1, j-1);
                uchar p8 = im.at<uchar>(i, j-1); 
            

                int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                         (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
                int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
                int N  = N1 < N2 ? N1 : N2;
                int m  = iter == 0 ? ((p6 | p7 | !p1) & p8) : ((p2 | p3 | !p5) & p4);

                if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
                    marker.at<uchar>(i,j) = 1;
                    diff = 1;
                }
            }
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinningGuoHall(Mat& inimg, Mat& outimg)
{
    int diff = 0;
    inimg.copyTo(outimg);
    do {
        diff = 0;
        thinningGuoHallIteration(outimg, 0, diff);
        thinningGuoHallIteration(outimg, 1, diff);
    } 
    while (diff != 0);    
}

/**
 * This is an example on how to call the thinning function above.
 */
int main(int argc, char const **argv)
{
    if(argc < 2)
    {
      cout << "Please input image!" << endl;
      return 0;
    }
    Mat inimg = cv::imread(argv[1], 0);
    if (inimg.empty())
        return -1;
    inimg /= 255;

    Mat outimg(inimg.size(), CV_8UC1);
    

    clock_t start,finish;
    double totaltime;
    start=clock();

    for (int i = 0; i < 100; i++)
    {
        thinningGuoHall(inimg, outimg);
    }

    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\n此程序的运行时间为"<<totaltime<<"ms！"<<endl;

    outimg *= 255;
    // cv::imshow("src", src);
    // cv::imshow("dst", outimg);
    // cv::waitKey();
    return 0;
}

// 1638 ms
// 1544
