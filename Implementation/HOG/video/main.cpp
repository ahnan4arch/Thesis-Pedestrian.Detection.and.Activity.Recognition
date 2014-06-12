#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame, const string currentImageFile );

String window_name = "Pedestrian detection";		//窗口ID
String input_video="input.avi";     //输入视频文件

/**
 * 显示检测结果
 * @param found:包含有效检测结果的矩形
 * @param imageData:原始测试图像
 */
static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
		  r.x += cvRound(r.width*0.1);
		  r.width = cvRound(r.width*0.8);
		  r.y += cvRound(r.height*0.07);
		  r.height = cvRound(r.height*0.8);
        rectangle(imageData, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		  putText(imageData,"Pedestrian",Point(r.x, r.y+r.height),
				  FONT_HERSHEY_SIMPLEX,r.width*0.006,CV_RGB(255, 20, 147),2.0);
    }
}

/**
 * 测试检测器
 * @param hog:HOG描述符
 * @param imageData：测试图像
 */
static void detectTest(const HOGDescriptor& hog, Mat& imageData) {
    vector<Rect> found;
    int groupThreshold = 2;
    Size padding(Size(32, 32));
    Size winStride(Size(8, 8));
    double hitThreshold = 0.; 
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, 1.05, groupThreshold);
    showDetections(found, imageData);
}


int main(int argc, char** argv) {
    Mat frame;
    HOGDescriptor hog; 
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    VideoCapture inputVideo(input_video);
    if( inputVideo.isOpened() ){
        double fps=inputVideo.get(CV_CAP_PROP_FPS);
        cout << "帧/秒：" << fps << endl;
		while(true)
        {
            bool bSuccess = inputVideo.read( frame );
            if( !bSuccess )
            {
                cout << "不能从视频文件读取帧" << endl;
                break;
            }
            detectTest(hog, frame);
            imshow(window_name, frame);
            if (waitKey(30)==27)
            {
                cout << "按下ESC键" << endl;
                break;
            }
        }
	}
    else{
		printf(" --(!) Error reading video -- Break!");
		return -1;
	}
	return EXIT_SUCCESS;
}

