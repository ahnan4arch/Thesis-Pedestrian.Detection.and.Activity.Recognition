#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

String cascade_name = "haarcascade_fullbody.xml";   //级联器文件
CascadeClassifier ped_cascade;			//级联器
string window_name = "Pedestrian detection";		//窗口ID

int main( int argc, const char** argv )
{
   Mat frame;

   //-- 1. 加载级联器
   if( !ped_cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. 读入测试文件
   frame = imread(argv[1]);

   //-- 3. 对测试文件进行检测
   if( !frame.empty() ){ 
		detectAndDisplay( frame ); 
	}
   else{ 
		printf(" --(!) Error reading image -- Break!"); 
		return -1; 
	}
	waitKey(0);
   return 0;
 }

void detectAndDisplay( Mat frame )
{
  std::vector<Rect> peds;	//目标位置
  Mat frame_gray;		//灰度图像

  cvtColor( frame, frame_gray, CV_BGR2GRAY );	//转换为灰度图像

  //-- 检测目标
  ped_cascade.detectMultiScale( frame_gray, peds, 1.1,1, 0|CV_HAAR_SCALE_IMAGE, Size(96, 96) );

  //-- 可视化标记
  for( size_t i = 0; i < peds.size(); i++ )
  {
    Point upleft( peds[i].x, peds[i].y );		//左上角
    Point downright( peds[i].x + peds[i].width, peds[i].y + peds[i].height );		//右下角
    rectangle( frame, upleft, downright, Scalar(255,255,0),2,CV_AA);	//矩形标记
  }
  //-- 可视化输出
  imshow( window_name, frame );

  //-- 写入输出文件
  imwrite( "output.jpg", frame );
}
