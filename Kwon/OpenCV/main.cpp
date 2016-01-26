/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;


void thinIter1(Mat& tempimg, Mat& outimg, int& flag)
{

    for (int i = 1; i < tempimg.rows-1; i++)
    {
        for (int j = 1; j < tempimg.cols-1; j++)
        {
            if(tempimg.at<uchar>(i,j) == 1)
            {
                uchar p2 = tempimg.at<uchar>(i-1, j);
                uchar p3 = tempimg.at<uchar>(i-1, j+1);
                uchar p4 = tempimg.at<uchar>(i, j+1);
                uchar p5 = tempimg.at<uchar>(i+1, j+1);
                uchar p6 = tempimg.at<uchar>(i+1, j);
                uchar p7 = tempimg.at<uchar>(i+1, j-1);
                uchar p8 = tempimg.at<uchar>(i, j-1);
                uchar p9 = tempimg.at<uchar>(i-1, j-1);

                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = (p2 * p4 * p6);
                int m2 = (p4 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0){
                    outimg.at<uchar>(i,j) = 0;
                    flag = 1;
                }
            }
        }
    }
}

void thinIter2(Mat& tempimg, Mat& outimg, int& flag)
{

    for (int i = 1; i < tempimg.rows-1; i++)
    {
        for (int j = 1; j < tempimg.cols-1; j++)
        {
            if(tempimg.at<uchar>(i,j) == 1)
            {
                uchar p2 = tempimg.at<uchar>(i-1, j);
                uchar p3 = tempimg.at<uchar>(i-1, j+1);
                uchar p4 = tempimg.at<uchar>(i, j+1);
                uchar p5 = tempimg.at<uchar>(i+1, j+1);
                uchar p6 = tempimg.at<uchar>(i+1, j);
                uchar p7 = tempimg.at<uchar>(i+1, j-1);
                uchar p8 = tempimg.at<uchar>(i, j-1);
                uchar p9 = tempimg.at<uchar>(i-1, j-1);

                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = (p2 * p4 * p8);
                int m2 = (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0){
                    outimg.at<uchar>(i,j) = 0;
                    flag = 1;
                }
            }
        }
    }
}

void thinningC2(Mat& tempimg, Mat& outimg)
{
    for (int i = 1; i < tempimg.rows-1; i++)
    {
        for (int j = 1; j < tempimg.cols-1; j++)
        {
            if(tempimg.at<uchar>(i,j) == 1){
                uchar p1 = tempimg.at<uchar>(i-1, j-1);
                uchar p2 = tempimg.at<uchar>(i-1, j);
                uchar p3 = tempimg.at<uchar>(i-1, j+1);
                uchar p4 = tempimg.at<uchar>(i, j+1);
                uchar p5 = tempimg.at<uchar>(i+1, j+1);
                uchar p6 = tempimg.at<uchar>(i+1, j);
                uchar p7 = tempimg.at<uchar>(i+1, j-1);
                uchar p8 = tempimg.at<uchar>(i, j-1);
                if((p3 == 0 && (p1*p8*p6==1 || p5*p6*p8==1)) || (p1 == 0 && (p2*p4*p6==1 || p4*p6*p7==1))){
                    outimg.at<uchar>(i,j) = 0;
                }
            }
        }
    }
}
/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(Mat& inimg, Mat& outimg)
{
    inimg.copyTo(outimg);
    outimg /= 255;

    int flag = 0;
    Mat tempimg(inimg.size(), CV_8UC1);
    
    do {
        flag = 0;
        outimg.copyTo(tempimg);
        thinIter1(tempimg, outimg, flag);
        outimg.copyTo(tempimg);
        thinIter2(tempimg, outimg, flag);
    } 
    while (flag > 0);
    outimg.copyTo(tempimg);
    thinningC2(tempimg, outimg);
    outimg *= 255;
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

    cv::Mat inimg = cv::imread(argv[1], CV_8UC1);
    if (inimg.empty())
        return -1;

    Mat outimg(inimg.size(), CV_8UC1);

    clock_t start,finish;
    double totaltime;

    start=clock();
    for (int i = 0; i < 100; ++i)
        thinning(inimg, outimg);
    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\nZhang-Sune_Thinning runTime = "<<totaltime <<"ms！"<<endl;

    string str(argv[1]);
    str = "out-" + str.substr(10);
    imwrite( str, outimg);
    
    cv::waitKey(0);

    return 0;
}
