#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std; 
using namespace cv;
cv::Mat rgb2gray(cv::Mat img){
	cv::Mat res(img.rows,img.cols,CV_8UC1,cvScalarAll(0));
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			res.at<uchar>(i,j)=(4*img.at<cv::Vec3b>(i,j)[0]+4*img.at<cv::Vec3b>(i,j)[1]+img.at<cv::Vec3b>(i,j)[2])/9;
		}
	}
	return res;
}



int main(int argc, char **argv){
	cv::Mat orig1=cv::imread(argv[1],1);
	cv::Mat orig ;
	orig=rgb2gray(orig1);
	
	cvtColor(orig1, orig, CV_BGR2GRAY);

	cv::namedWindow("image",0);


	//Does the median blur
	cv::medianBlur(orig, orig, 5);
	

	// Gaussian Meadian Blur
	float dst[255] ;



	cv::threshold(orig, orig, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	cv::imwrite(argv[2], orig);
}



//********************************************************************************


