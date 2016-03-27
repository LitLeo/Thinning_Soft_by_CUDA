#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

#define HIGH 255
#define LOW 0
void printMat(Mat im)
{
    //im /= 255;
    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            cout << (int)im.at<uchar>(i,j) / 255 << " ";
        }
        cout << endl;
    }

    cout << endl << endl << endl;
}

void thin(Mat& inimg, Mat& outimg)
{
    Mat tempimg;
    inimg.copyTo(outimg);

    int flag = 1;
    while (flag) {
        flag = 0;

        outimg.copyTo(tempimg);
        for (int r = 1; r < outimg.rows - 1; r++) 
            for (int c = 1; c < outimg.cols - 1; c++) 
                
                if (tempimg.at<uchar>(r, c) == HIGH){
                    uchar x1 = tempimg.at<uchar>(r - 1, c - 1);
                    uchar x2 = tempimg.at<uchar>(r - 1, c);
                    uchar x3 = tempimg.at<uchar>(r - 1, c + 1);
                    uchar x4 = tempimg.at<uchar>(r, c - 1);
                    uchar x5 = tempimg.at<uchar>(r, c + 1);
                    uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                    uchar x7 = tempimg.at<uchar>(r + 1, c);
                    uchar x8 = tempimg.at<uchar>(r + 1, c + 1); 
                    
                    if ((x4 == HIGH && x5 == LOW) &&
                        !((x2 == LOW && x3 == HIGH) || (x7 == LOW && x8 == HIGH)) && 
                        !(x1 == LOW && x2 == LOW && x3 == LOW && x6 == LOW && x7 == LOW && x8 == LOW)){
                        outimg.at<uchar>(r,c) = LOW;
                        flag = 1;
                        //continue;
                    }
                }

        outimg.copyTo(tempimg);
        for (int r = 1; r < outimg.rows - 1; r++) 
            for (int c = 1; c < outimg.cols - 1; c++) 
                
                if (tempimg.at<uchar>(r, c) == HIGH){
                    uchar x1 = tempimg.at<uchar>(r - 1, c - 1);
                    uchar x2 = tempimg.at<uchar>(r - 1, c);
                    uchar x3 = tempimg.at<uchar>(r - 1, c + 1);
                    uchar x4 = tempimg.at<uchar>(r, c - 1);
                    uchar x5 = tempimg.at<uchar>(r, c + 1);
                    uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                    uchar x7 = tempimg.at<uchar>(r + 1, c);
                    uchar x8 = tempimg.at<uchar>(r + 1, c + 1); 
                    
                    if (x7 == HIGH && x2 == LOW && 
                        !((x4 == LOW && x1 == HIGH) || (x5 == LOW && x3 == HIGH)) && 
                        !(x1 == LOW && x4 == LOW && x6 == LOW && x3 == LOW && x5 == LOW && x8 == LOW)){
                        outimg.at<uchar>(r,c) = LOW;
                        flag = 1;
                        //continue;
                    }
                }
        
        outimg.copyTo(tempimg);
        for (int r = 1; r < outimg.rows - 1; r++) 
            for (int c = 1; c < outimg.cols - 1; c++) 
                
                if (tempimg.at<uchar>(r, c) == HIGH){
                    uchar x1 = tempimg.at<uchar>(r - 1, c - 1);
                    uchar x2 = tempimg.at<uchar>(r - 1, c);
                    uchar x3 = tempimg.at<uchar>(r - 1, c + 1);
                    uchar x4 = tempimg.at<uchar>(r, c - 1);
                    uchar x5 = tempimg.at<uchar>(r, c + 1);
                    uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                    uchar x7 = tempimg.at<uchar>(r + 1, c);
                    uchar x8 = tempimg.at<uchar>(r + 1, c + 1); 
                    
                    if (x5 == HIGH && x4 == LOW && 
                        !((x2 == LOW && x1 == HIGH) || (x7 == LOW && x6 == HIGH)) &&
                        !(x1 == LOW && x2 == LOW && x3 == LOW && x6 == LOW && x7 == LOW && x8 == LOW)){
                        outimg.at<uchar>(r,c) = LOW;
                        flag = 1;
                        //continue;
                    }
                }

        outimg.copyTo(tempimg);
        for (int r = 1; r < outimg.rows - 1; r++) 
            for (int c = 1; c < outimg.cols - 1; c++) 
                
                if (tempimg.at<uchar>(r, c) == HIGH){
                    uchar x1 = tempimg.at<uchar>(r - 1, c - 1);
                    uchar x2 = tempimg.at<uchar>(r - 1, c);
                    uchar x3 = tempimg.at<uchar>(r - 1, c + 1);
                    uchar x4 = tempimg.at<uchar>(r, c - 1);
                    uchar x5 = tempimg.at<uchar>(r, c + 1);
                    uchar x6 = tempimg.at<uchar>(r + 1, c - 1);
                    uchar x7 = tempimg.at<uchar>(r + 1, c);
                    uchar x8 = tempimg.at<uchar>(r + 1, c + 1); 
                    
                    if (x2 == HIGH && x7 == LOW && 
                        !((x4 == LOW && x6 == HIGH) || (x5 == LOW && x8 == HIGH)) && 
                        !(x1 == LOW && x4 == LOW && x6 == LOW && x3 == LOW && x5 == LOW && x8 == LOW)){
                        outimg.at<uchar>(r,c) = LOW;
                        flag = 1;
                        //continue;
                    }
                }

        
    }
}

int main(int argc, const char** argv)
{
    if(argc < 2)
    {
      cout << "Please input image!" << endl;
      return 0;
    }

    Mat inimg = imread(argv[1], CV_8UC1);
    Mat outimg(inimg.size(), CV_8UC1);
    clock_t start,finish;
    double totaltime;
    start=clock();

    for (int i = 0; i < 100; ++i)
        thin(inimg, outimg);
        
    finish=clock();
    totaltime=(double)(finish-start)*10/CLOCKS_PER_SEC;
    cout<<"\nrun time = "<<totaltime<<"ms！"<<endl;

    // imshow("outimg", outimg);
    imwrite("outimg.bmp", outimg);
    // waitKey();
    return 0;
    return 0;
}