#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  
#include <sys/stat.h>  
#include <string.h> 




/*********************************************************************************************
 * compile with:
 * g++ -O3 -o gesture gesture.cpp -std=c++11 `pkg-config --cflags --libs opencv`
*********************************************************************************************/


/*
 *                                SPECIFICATION
 *  The programe aims to recognise gesture in an image. You can specify a static image
 *  for recognising, or capture image via camera without specifing any parameter.
 *  The processing workflow is
 *  #1 If there is a saved MLP 'bp.xml' under the same dir, load it
 *  #2 If there is no 'bp.xml' under the same dir, train the MLP
 *  #3 Preprocess the static image or frame from camera by following steps:
 *      1) convert the image to gray
 *      2) threshold, filter all pixels whose grey scale is larger than 230 or small than 70 to
 *         avoid the affection by background
 *      3) Median filter to avoid affection by noise
 *      4) Threshold with specifing CV_THRESH_OTSU to extract the gesture
 *      5) Find the largest contour
 *      6) Calculate the CEs from the contour
 *      7) Predict the result via the MLP
 *  #4 Show result 
 * 
 * 
 *                                   NOTE
 *  #1 To train the MLP, there should be a folder named "trainImgs" under the same directory
 *     with the executable. All train images should putted be in the folder "trainImgs".
 *  #2 All names of all train images should contain at least 7 characters, and the 7th
 *     char should be the gesture number, for example, the train image
 *     "hand1_0_bot_seg_1_cropped.png" means the gesture of this image representes '0'.
 *  #3 The train process may take a few seconds to complete.
 *  #4 To recognise gesture via camera, you should make sure the background of the hand
 *     is black(or the grey scale is very different with the hand), so that the gesture
 *     could be extracted correctly.
 */


using namespace std;
using namespace cv;
using namespace chrono;

#define Mpixel(image,x,y) (( uchar *)(((image).data)+(y)*((image).step)))[(x)]

#define PI 3.141592653
#define CE_COUNT 20 // Calculate 20 CEs for a contour
#define CLASSIFIER_COUNT 10 // 10 classifiers

char trainDir[100] = "./trainImgs";  
int const MAX_STR_LEN = 200;  
CvANN_MLP bp;
    
  
/* Store all train image names to the vector imgs_names*/  
bool getAllTrainImgs( const char *dir_name, vector<string> &imgs_names)  
{  
    // check the parameter !  
    if( NULL == dir_name )  
    {  
        cout<<" dir_name is null ! "<<endl;  
        return false;  
    }  
    // check if dir_name is a valid dir  
    struct stat s;  
    lstat( dir_name , &s );  
    if( ! S_ISDIR( s.st_mode ) )  
    {  
        cout<<"dir_name is not a valid directory !"<<endl;  
        return false;  
    }  
    struct dirent * filename;    // return value for readdir()  
    DIR * dir;                   // return value for opendir()  
    dir = opendir( dir_name );  
    if( NULL == dir )  
    {  
        cout<<"Can not open dir "<<dir_name<<endl;  
        return false;  
    }  
    /* read all the files in the dir */  
    while( ( filename = readdir(dir) ) != NULL )  
    {  
        // get rid of "." and ".."  
        if( strcmp( filename->d_name , "." ) == 0 ||   
            strcmp( filename->d_name , "..") == 0    )  
            continue;
        imgs_names.push_back(filename->d_name);
    } 
    return true;
}   
  
// Calculate 20 CEs from a contour
  
void EllipticFourierDescriptors(vector<Point>& contour,vector<float> &CE){
    vector<float>ax,ay,bx,by;
    int m=contour.size();
    int n=20;
    float t = (2*PI)/m;
    for(int k=0;k<n;k++){
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);
        for(int i=0;i<m;i++)
        {
            ax[k] = ax[k] + contour[i].x * cos((k+1) * t * (i));
            bx[k] = bx[k] + contour[i].x * sin((k+1) * t * (i));
            ay[k] = ay[k] + contour[i].y * cos((k+1) * t * (i));
            by[k] = by[k] + contour[i].y * sin((k+1) * t * (i));
        }
        ax[k]=(ax[k])/m;
        bx[k]=(bx[k])/m;
        ay[k]=(ay[k])/m;
        by[k]=(by[k])/m;
    }
    for(int k=0;k<n;k++){
        CE.push_back(sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(
        ax[0]*ax[0]+ay[0]*ay[0]))+sqrt((bx[k]*bx[k]+by[k]*by[k])/(
        bx[0]*bx[0]+by[0]*by[0])));
    }
}


// Get CEs from an image
bool getTrainCEs(Mat image, vector<float> &CEs)
{
    cvtColor(image,image,CV_BGR2GRAY);
    /*
     * The train image may have a different backgrounds, we need to turn the background
     * to be black firstly. So set every pixel to be 0 if its gray scale > 230
     * or < 80;
     */ 
    threshold(image,image,70,255,CV_THRESH_TOZERO);
    threshold(image,image,230,255,CV_THRESH_TOZERO_INV);
    // Filter noise
    medianBlur(image, image, 7);
    threshold(image, image, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
    vector<vector<Point>> contours;
    Mat canny_output;
    vector<Vec4i> hierarchy;
    findContours(image,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    int largestcontour=0;
    long int largestsize=0;
    for(int i=0;i<contours.size();i++)
    {
        if(largestsize<contours[i].size()){
            largestsize=contours[i].size();
            largestcontour=i;
        }
    }
    if(largestsize <= 10)
    {
        cout<< "Error: Countour size is too small" << endl;
        return false;
    }
    EllipticFourierDescriptors(contours[largestcontour],CEs);
    return true;
}


// Generate train data and label
bool generateTrainData(vector<vector<float>> &data, vector<vector<float>> &label)
{
    vector<string> trainImgs;
    getAllTrainImgs(trainDir, trainImgs);
    
    
    int count = trainImgs.size();
    
    if(count <= 0)
    {
        cout<<"ERROR: no train image found"<<endl;
        return false;
    }
    for(int i = 0; i < count; i ++)
    {
        int value;
        const char *imgName = trainImgs[i].c_str();
        /*
         * The name of traing image should like "hand3_3_dif_seg_3_cropped.png"
         * The 7th char is the class number.
         */ 
        if(strlen(imgName) < 7)
            continue;
        value = *(imgName+6) - '0';
        if(value < 0 || value > 9)
            continue;
        vector<float> lableData;
        
        for(int i = 0; i < CLASSIFIER_COUNT; i ++)
            lableData.push_back(0.0);
        lableData[value] = 1.0;
        vector<float> CEs;
        // Generate image directory and read it
        char imgDir[200] = {0};
        strcat(imgDir, trainDir);
        strcat(imgDir, "/");
        strcat(imgDir, imgName);
        Mat src = imread(imgDir);
        if(getTrainCEs(src, CEs) && CEs.size() == CE_COUNT)
        {
            // Save it for training
            label.push_back(lableData);
            data.push_back(CEs);
        }
    }
    return true;
}


// Train the MLP, save the MLP as "bp.xml" 

bool train()
{
    vector<vector<float>> data;
    vector<vector<float>> label;
    
    if(!generateTrainData(data, label))
        return false;
    
    // Conver vector to array
    int trainDatacount = data.size();
    if(trainDatacount <= 0)
        return false;
    float* dataArray = new float[trainDatacount * CE_COUNT];
    float* tempda = dataArray;
    
    float* labelArray = new float[trainDatacount * CLASSIFIER_COUNT];
    float* templa = labelArray;
    
    for(int i = 0; i < trainDatacount; i ++)
        for(int j = 0; j < CE_COUNT; j ++)
            *(tempda++) = data[i][j];

    for(int i = 0; i < trainDatacount; i ++)
        for(int j = 0; j < CLASSIFIER_COUNT; j ++)
            *(templa++) = label[i][j];
    Mat trainData(trainDatacount, CE_COUNT, CV_32FC1, dataArray);
    Mat trainLabel(trainDatacount, CLASSIFIER_COUNT, CV_32FC1, labelArray);

    CvANN_MLP_TrainParams params;
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.001;    
    params.bp_moment_scale = 0.1;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001);     

    Mat layerSizes = (Mat_<int>(1, 4) << 20, 20, 20, 10);
 
    // Save the MLP future prediction
    bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
    bp.train(trainData, trainLabel, Mat(), Mat(), params);
    bp.save("bp.xml");
    
    // Clear memory
    delete[] dataArray;
    delete[] labelArray;
    
    for(int i = 0; i < data.size(); i ++)
        data[i].clear();
    data.clear();
    for(int i = 0; i < label.size(); i ++)
        label[i].clear();
    label.clear();
}



// Recofnise gesture from an image.
int recogGesture(Mat image, Mat &threshImage, Mat &contourImage)
{
    cvtColor(image,threshImage,CV_BGR2GRAY);
    /*
     * The test image may have a very bright background, and the sleeve under
     * the hand is black, for applying a threshold, we need to turn the background
     * to be black firstly. So set every pixel to be 0 if its gray scale > 230
     * or < 70;
     */
    threshold(threshImage,threshImage,230,255,CV_THRESH_TOZERO_INV);
    threshold(threshImage,threshImage,70,255,CV_THRESH_TOZERO);
    // Median blur to filter noise
    medianBlur(threshImage, threshImage, 7);
    Mat midImage = threshImage.clone();
    threshold(midImage,midImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);

    //Find out the largest contour
    vector<vector<Point>> contours;
    Mat canny_output;
    vector<Vec4i> hierarchy;
    findContours(midImage,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int largestcontour=0;
    long int largestsize=0;
    for(int i=0;i<contours.size();i++)
    {
        if(largestsize<contours[i].size()){
            largestsize=contours[i].size();
            largestcontour=i;
        }
    }
    if(largestsize <= 20)
    {
        cout<< "Error: Countour size is too small" << endl;
        return -1;
    }
    // calculate CEs and predict via MLP
    vector<float>CE;
    EllipticFourierDescriptors(contours[largestcontour],CE);
                
    float *dataArray = &CE[0];
    Mat sampleMat(1, CE_COUNT, CV_32FC1, dataArray);
    Mat responseMat;
    bp.predict(sampleMat, responseMat);
    float* p=responseMat.ptr<float>(0);
    int result = 0;
    float maxLabel = 0;
    for(int i = 0; i < CLASSIFIER_COUNT; i ++)
    {
        if(maxLabel < *(p + i))
        {
            maxLabel = *(p + i);
            result = i;
        }
    }
    // Draw the contour for user.
    contourImage=Mat::zeros(threshImage.size(),CV_8UC3);
    Scalar color=CV_RGB(0,255,0);
    drawContours(contourImage,contours,largestcontour,color,1,8);

    char str[25];
    snprintf(str, sizeof(str), "gesture:%d", result);
    putText(image,str,cvPoint(10, 60),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(255,128,128));
    return result;
}


// Recognise gesture via camera 
void showCam()
{
    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        cout << "Failed to open camera" << endl;
        return;
    }
    cout << "Opened camera" << endl;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, threshImage, contourImage;
    cap >> frame;
    printf("frame size %d %d \n",frame.rows, frame.cols);
    int key=0;
    double fps=0.0;
    namedWindow("step1: Input Image",WINDOW_AUTOSIZE);
    namedWindow("step2: Extract and median filter",WINDOW_AUTOSIZE);
    namedWindow("step3: Contours",WINDOW_AUTOSIZE);
    while (true)
    {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if( frame.empty() )
            break;
        int result = recogGesture(frame, threshImage, contourImage);
        
        system_clock::time_point end = system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        fps = 1000000/seconds;
        cout << "frames: " << fps << " seconds: " << seconds << " gesture: "<< result << endl;
        char printit[100];
        sprintf(printit,"frames: %2.1f",fps);
        putText(frame, printit, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(128,128,128), 2, 8);
        imshow("step1: Input Image",frame);
        imshow("step2: Extract and median filter", threshImage);
        imshow("step3: Contours",contourImage);
        key=waitKey(5);
        if(key==113 || key==27)
            return;//either esc or 'q'
    }
}


// Check if file exists

inline bool exists_test(const string name) {
    ifstream f(name.c_str());
    return f.good();
}


// Main entry
int main(int argc,char**argv)
{
    // Train the MLP or load the MLP if it has been trained
    if(exists_test("bp.xml"))
    {
        bp.load("./bp.xml");
    } else {
        cout<<"No bp.xml found, MLP starts training, this may take a few seconds..."<<endl;
        if(!train())
        {
            cout<<"Unable to train MLP, please make sure there are train images under ./trainImgs." << endl;
            cout<<"The 7th char of the train image name should be the class number, for example:"<<endl;
            cout<<"hand1_2.png means it is gesture 2"<< endl;
            return false;
        }
        cout<<"MLP training finished"<<endl;
    }
    if ( argc != 2)
    {
        // Show camera if user did not specify a static image
        showCam();
        return 0;
    }
    // Create 3 windows to show the process
    namedWindow("step1: Input Image",WINDOW_AUTOSIZE);
    namedWindow("step2: Extract and median filter",WINDOW_AUTOSIZE);
    namedWindow("step3: Contours",WINDOW_AUTOSIZE);
    /*
     * If user specifies a static test image, process it.
     */
    Mat image, threshImage, contourImage;
    image=imread(argv[1]);
    if(!image.data)
    {
        cout<<"read image error: "<<argv[1]<<endl;
        return 0;
    }
    int result = recogGesture(image, threshImage, contourImage);
    if(result != -1)
        cout << "The gesture has been classified as " << result << endl;
    imshow("step1: Input Image",image);
    imshow("step2: Extract and median filter", threshImage);
    imshow("step3: Contours",contourImage);
    waitKey(0);
}
