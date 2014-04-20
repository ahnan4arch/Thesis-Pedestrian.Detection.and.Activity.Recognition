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
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
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
	// 加载训练的HOG特征描述符文件
    hog.setSVMDetector(descriptorVector); 
    Mat testImage;
	 testImage=imread("test.jpg");
	 imshow("origin image",testImage);
    detectTest(hog, testImage);
    imshow("HOG custom detection", testImage);
	 waitKey(0);
	 return EXIT_SUCCESS;
}
