/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

#define HIGH 255
#define LOW 0


/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& im, int iter, int& diff)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            if(im.at<uchar>(i,j) == 1) {
                uchar p2 = im.at<uchar>(i-1, j);
                uchar p3 = im.at<uchar>(i-1, j+1);
                uchar p4 = im.at<uchar>(i, j+1);
                uchar p5 = im.at<uchar>(i+1, j+1);
                uchar p6 = im.at<uchar>(i+1, j);
                uchar p7 = im.at<uchar>(i+1, j-1);
                uchar p8 = im.at<uchar>(i, j-1);
                uchar p9 = im.at<uchar>(i-1, j-1);

                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0){
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
void thinning(Mat& inimg, Mat& outimg)
{
    int diff = 0;
    inimg.copyTo(outimg);
    do {
        // cout << "diff = " << diff << endl;
        diff = 0;
        thinningIteration(outimg, 0, diff);
        thinningIteration(outimg, 1, diff);
    } 
    while (diff != 0); 
}

/**
 * This is an example on how to call the thinning function above.
 */
int main(int argc, const char **argv)
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
        thinning(inimg, outimg);
    }

    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\nZhang-Sune_Thinning runTime = "<<totaltime <<"ms！"<<endl;

    string str(argv[1]);
    str = "out-" + str.substr(10);
    imwrite( str, outimg * 255);
    
    cv::waitKey(0);

    return 0;
}
