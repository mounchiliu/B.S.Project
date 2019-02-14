#ifndef FRAME_H
#define FRAME_H

#include<vector>
#include <opencv2/opencv.hpp>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
//#include "Thirdparty/DBow3/src/BowVector.h"
//#include "Thirdparty/DBow3/src/FeatureVector.h"


#include "ORBextractor.h"
#include "BOWVocabulary.h"

#include "MapPoint.h"
#include "KeyFrame.h"

using namespace std;

namespace ORB_SLAM
{

#define GRID_ROWS 48
#define GRID_COLS 64

class MapPoint;
class KeyFrame;
class Frame
{
public:

   Frame(){}
   //Copy constructor
   Frame(const Frame &frame);

   //Constructor for RGB-D cameras.
   Frame(const cv::Mat &image_Gray, const cv::Mat &image_Depth, const double &timeStamp,
         ORBextractor* pextractor, ORBVocabulary* pVocabulary, cv::Mat &CalibrationMatrix, cv::Mat &distort,
         const float &bfx, const float &thDepth);
   //Function to extract ORB on the image. 0 for left image and 1 for right image.
   void ExtractORB(const cv::Mat &image);

   //For each frame, compute the bag of words
   void FindBOW();

   //Update the camera pose.
   void UpdatePose(cv::Mat TWorld2Cam);

   //Computes matrices for rotation and translation to transform between World coordinates and camera coordinates
   void PoseMatrices();

   //Get camera center
   cv::Mat GetCameraCentre();

   //Get inverse Rotation Matrix
   cv::Mat GetInverseRotation();

   //Compute the position of keypoint in cell
   //Figure out whether the keypoint is in a grid or not
   bool InGrid(int &pos_X, int &pos_Y);

   //return the index of the KeyPoint which satisfies the requirement
   //x,y -> coordinate of the KeyPoint in current frame
   //this coordinate will be campared with the keypoints in reference frame using method below
   //r   -> length of the searching area
   //minLevelOfPyramid & maxLevelOfPyramid limit the level of the pyramid from which the KP is extracted
   vector<size_t> FeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevelOfPyramid=-1, const int maxLevelOfPyramid=-1) const;

   //Associate a coordinate to a keypoint if there is valid depth i the depth map
   void ComputeStereoFromRGBD(const cv::Mat &image_Depth);

   //Backproject a keypoint into 3D world coordinates
   cv::Mat BackProject(const int &i);

   bool MPInViewArea(MapPoint* pMP);


   //data member

   // Feature extractor.
   ORBextractor* mpORBextractor;

   //Vocabulary
   ORBVocabulary* mpVocabulary;
   //DBoW3::Vocabulary* mpVocabulary;


   // Frame timestamp.
   double mTimeStamp;

   //Next Frame ID
   static long unsigned int LastId;
   //Current Frame ID
   long unsigned int mnId;

   // Calibration matrix and distortion parameters.
   cv::Mat mCalibrationMatrix;
   static float fx;
   static float fy;
   static float cx;
   static float cy;
   static float invfx;
   static float invfy;
   cv::Mat mDistortCoef;
\



   // Threshold close/far threshold
   float mThreshDepth;

   //double of keypoints
   int mNumKeypoints;

   // Vector of keypoints (original) and undistorted keypoints(used by the system).
   // RGB images can be distorted.
   vector<cv::KeyPoint> mvKeyPoints;
   vector<cv::KeyPoint> mvUndisKP;

   //Keypoint corresponds to depth
   vector<float> mvDepthWithKPU;//The KP is associated with depth INFO
   //Depth INFO
   vector<float> mvDepth;

   //relative MP to the KP
   vector<MapPoint*> mvMapPoint;
   //MP outside the area
   vector<bool> mvOutliers;

   // Baseline multiplied by fx.
   float mbfx;
   // Baseline in meters.
   float mb;

   // ORB descriptor
   // each row holds one descriptor for one keypoint.
   cv::Mat mDescriptors;
   //BOW INFO
   DBoW2::BowVector mBOWVector;
   DBoW2::FeatureVector mFVector;



    //keyPoints are assigned to cell in a grid uniformly to reduce the complexity of matching
    //Divide the image to grib to ensure the uniform extraction
    //The coordinate values multiply with the numbers below respectively to get which grid the point will be.
    static float mfGridIndividualWidth;
    static float mfGridIndividualHeight;
    vector<size_t> mvGrid[GRID_COLS][GRID_ROWS];


    //Transform matrix to transfer from World to camera
    cv::Mat mTWorld2Cam;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
   // float mfLogScaleFactor;
    vector<float> mvInvScaleFactors;
    vector<float> mvScaleFactors;



    //Undistorted Image Bounds.
    static float mnMin_X;
    static float mnMax_X;
    static float mnMin_Y;
    static float mnMax_Y;

    //Check initialization
    static bool mbInitialComputations;


    //Reference frame
    KeyFrame* mpReferenceFrame;


private:

    vector<cv::Mat> DescriptorVector(const cv::Mat &Descriptors);

    //Use opencv Library to remove the distortion in keypoints
    void UndistortKeyPoints();

    //Compute image bounds for undistorted image
    void ComputeImageBounds(const cv::Mat &image);

    //Assign keypoints to the grid
    void AssignFeaturesToGrid();

    //Rotation matrix for transformation from world coordinate to camera coordinate
    cv::Mat mRWorld2Cam;
    //Translation matrix for transformation from world to camera
    cv::Mat mtWorld2Cam;
    //Rotation matrix for transformation from camera to world
    cv::Mat mRCam2World;
    //element in Translation matrix for transformation from camera to world
    //-R^(T)*t
    cv::Mat mtCam2world;


};
}// end of namespace

#endif
