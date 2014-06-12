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

static void getFilesInDirectory(const string& dirName,
        vector<string>& fileNames, const vector<string>& validExtensions);

static string testImagesFolder = "test/";

// 字母小写转换
static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}
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
    static vector<string> testImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("pgm");
    getFilesInDirectory(testImagesFolder, testImages, validExtensions);
    unsigned long overallSamples = testImages.size();
    cout << "总共：" << overallSamples << "个文件" << endl;
	 // 使用默认参数
    HOGDescriptor hog; 
    // DT-2005建议参数
	 //hog.winSize = Size(64, 128); 
	 // 加载训练的HOG特征描述符文件
    //hog.setSVMDetector(descriptorVector); 
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    for(unsigned long tmp = 0; tmp < overallSamples; ++tmp){
        const string currentImageFile = testImages.at(tmp);
        //-- 2. 读入测试文件
        frame = imread(currentImageFile);
        //-- 3. 对测试文件进行检测
        if( !frame.empty() ){
            detectTest(hog, frame);
            imwrite(currentImageFile,frame);
        }
        else{
		printf(" --(!) Error reading image -- Break!");
		return -1;
        }
	}
	return EXIT_SUCCESS;
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
