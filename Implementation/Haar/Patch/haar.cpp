#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame, const string currentImageFile );

static void getFilesInDirectory(const string& dirName,
        vector<string>& fileNames, const vector<string>& validExtensions);

String cascade_name = "haarcascade_fullbody.xml";   //级联器文件
CascadeClassifier ped_cascade;			//级联器
static string testImagesFolder = "test/";

// 字母小写转换
static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

int main( int argc, const char** argv )
{
    Mat frame;
    static vector<string> testImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("pgm");
    getFilesInDirectory(testImagesFolder, testImages, validExtensions);
    unsigned long overallSamples = testImages.size();
    cout << "总共：" << overallSamples << "个文件" << endl;

    //-- 1. 加载级联器
    if( !ped_cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    for(unsigned long tmp = 0; tmp < overallSamples; ++tmp){
        const string currentImageFile = testImages.at(tmp);
        //-- 2. 读入测试文件
        frame = imread(currentImageFile);
        //-- 3. 对测试文件进行检测
        if( !frame.empty() ){
		    detectAndDisplay( frame, currentImageFile );
        }
        else{
		printf(" --(!) Error reading image -- Break!");
		return -1;
        }
	}
    return 0;
}

void detectAndDisplay( Mat frame ,const string  currentImageFile)
{
  std::vector<Rect> peds;	//目标位置
  Mat frame_gray;		//灰度图像

  cvtColor( frame, frame_gray, CV_BGR2GRAY );	//转换为灰度图像

  //-- 检测目标
  ped_cascade.detectMultiScale( frame_gray, peds, 1.1,1, 0|CV_HAAR_DO_CANNY_PRUNING, Size(70, 70));

  //-- 可视化标记
  for( size_t i = 0; i < peds.size(); i++ )
  {
    Point upleft( peds[i].x, peds[i].y );		//左上角
    Point downright( peds[i].x + peds[i].width, peds[i].y + peds[i].height );		//右下角
    rectangle( frame, upleft, downright, Scalar(255,255,0),2,CV_AA);	//矩形标记
  }

  //-- 写入输出文件
  imwrite( currentImageFile, frame );
}

/**
 * 列出给定目录的所有文件，并返回字符串数组(路径+文件名)
 * @param dirName: 目录名
 * @param fileNames: 给定目录中找到的文件名
 * @param validExtensions: 有效文件后缀规定
 */
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("扫描样本目录 %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
				// 后缀位置
            extensionLocation = string(ep->d_name).find_last_of("."); 
            
				// 检查后缀
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("有效文件： '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("无效文件，跳过: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("打开目录遇到错误 '%s'!\n", dirName.c_str());
    }
    return;
}
