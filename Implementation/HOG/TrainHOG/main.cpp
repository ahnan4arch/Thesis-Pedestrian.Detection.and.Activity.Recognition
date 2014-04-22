#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "svmlight/svmlight.h"

using namespace std;
using namespace cv;

// 样本路径名
static string posSamplesDir = "pos/";
static string negSamplesDir = "neg/";

// 存储HOG特征
static string featuresFile = "features.dat";

// 存储SVM模型
static string svmModelFile = "svmlightmodel.dat";

// 存储HOG特征描述符
static string descriptorVectorFile = "descriptorvector.dat";

static const Size trainingPadding = Size(0, 0); //填充值
static const Size winStride = Size(8, 8);			//窗口步进

// 字母小写转换
static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

// 用于进度可视化
static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

/**
 * 保存给定的HOG特征描述符到文件
 * @param descriptorVector: 待保存的HOG特征描述符矢量
 * @param _vectorIndices
 * @param fileName
 */
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    printf("保存HOG特征描述符：'%s'\n", fileName.c_str());
    string separator = " "; // 特征分隔符
    fstream File;
    File.open(fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        for (int feature = 0; feature < descriptorVector.size(); ++feature) 
            File << descriptorVector.at(feature) << separator;	//写入特征并设置分隔符号
        File << endl;
        File.flush();
        File.close();
    }
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

/**
 * 从输入图像计算HOG特征描述符矢量
 * @param imageFilename：输入图像路径  
 * @param descriptorVector: 返回HOG特征描述符矢量
 * @param hog: 包含期望参数设置的HOG描述符
 */
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
    Mat imageData = imread(imageFilename, 0);
    if (imageData.empty()) {	//无效图像
        featureVector.clear();
        printf("无有效图像，跳出特征提取\n", imageFilename.c_str());
        return;
    }
    // 检测图像尺寸
    //if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
    //    featureVector.clear();
    //    printf("图像'%s' 尺寸(%u x %u)与HOG窗口(%u x %u)不匹配\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
    //    return;
    //}
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); 
}


int main(int argc, char** argv) {
	 // 使用默认参数
    HOGDescriptor hog; 
    // DT-2005建议参数
	 hog.winSize = Size(64, 128); 
    // 加载训练样本
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("pgm");
    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    // 检查样本数量
    if (overallSamples == 0) {
        printf("未找到有效样本\n");
        return EXIT_SUCCESS;
    }

    setlocale(LC_ALL, "C"); 
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;
    /**
	  * 以兼容SVMLight的格式存储描述符矢量文件,递交给SVM训练
     */ 
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
            
				// 显示进度
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
				// 计算HOG特征描述符矢量
            calculateFeaturesFromInput(currentImageFile, featureVector, hog);
            if (!featureVector.empty()) {
	
  					 //用+1标记阳性，-1标记阴性
                File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    File << " " << (feature + 1) << ":" << featureVector.at(feature);
                }
                File << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("打开文件遇到错误'%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }

	 //  训练SVM
    printf("调用SVMlight\n");
    SVMlight::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
    SVMlight::getInstance()->train(); // 训练
    printf("训练结束，保存SVM模型\n");
    SVMlight::getInstance()->saveModelToFile(svmModelFile);

    printf("SVMlight生成单个HOG特征矢量\n");
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    SVMlight::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    
	 return EXIT_SUCCESS;
}
