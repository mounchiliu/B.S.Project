#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"
#include "BOWVocabulary.h"

#include <mutex>

using namespace std;

namespace ORB_SLAM
{

class Frame;
class Map;
class MapPoint;


class KeyFrame{
public:
  //keyframe is built based on frame
  KeyFrame(Frame &frame, Map *pMap);

  //For each frame, compute the bag of words
  void FindBOW();

  //Update the camera pose.
  void UpdatePose(const cv::Mat &TWorld2Cam);

  //Get pose of the camera (Transformation Matrix)
  cv::Mat GetPose();

  //Get inverse pose (inverse transformation Matrix)
  cv::Mat GetInversePose();

  //Get camera center in world coordination
  cv::Mat CameraCentre();

  //Get the camera centre point for two-camera sensor
  cv::Mat StereoCentre();

  //Get Rotation Matrix
  cv::Mat GetRotationMatrix();

  //Get Translation Matrix
  cv::Mat GetTranslationMatrix();

  //Get MPs
  vector<MapPoint*> GetAllMapPoints();

  //Get Specific MP (LOCK)
  MapPoint* GetMP(size_t &ID);

  //Replace MP
  void ReplaceMPWith(size_t &ID, MapPoint* pMP);

  //add MP to KF
  void AddMapPointToKF(MapPoint* pMP, size_t &ID_KP);

  //Erase MP in KF
  void DeleteMPinKF(size_t &index);
  void DeleteMPinKF(MapPoint* pMP);

  //Get State of KF
  bool IsBad();

  //Set BAD flag
  void SetBad();

  //Get the number of good tracked points
  int GoodTrackedMPs(int &minObs);


  //For building and updating Connections
  //Add/Update connections between KFs
  void UpdateConnection();
  //Method for building connections
  void BuildConnection(KeyFrame* pKF, int &weight);
  //Erase Connection
  void EraseConnection(KeyFrame* pKF);
  //Update KFs high weight
  void UpdateBestCommonViewKFs();
  //Get KFs with high weight
  vector<KeyFrame*> GetBestCommonViewKFs(int n);
  //Get All connected KFs
  vector<KeyFrame*> GetConnectedKFs();
  //Get weight
  int GetWeightOfKF(KeyFrame* pKF);
  //Get Father
  KeyFrame* GetFather();
  //Change Father
  void ChangeFather(KeyFrame* pKF);


  //For the building of the Tree
  void AddChild(KeyFrame* pKF);
  void DeleteChild(KeyFrame* pKF);
  set<KeyFrame*> GetChildren();


  //Method for KP
  //Similar to Frame class
  vector<size_t> FeaturesInArea(const float &x, const float  &y, const float  &r) const;
  //Backproject a keypoint into 3D world coordinates
  cv::Mat BackProject(const int &i);

  //Similar to Frame class
  //Compute the position of keypoint in cell
  //Figure out whether the keypoint is in a grid or not
  bool InImage(float &posX, float &posY);

  //For info (attributes) of KeyFrame object

  //Current KeyFrame ID
  long unsigned int mnID;

  //ID for previous KeyFrame
  static long unsigned int mnLastID;

  //Record the ID for the frame used to initialize this KeyFrame
  const long unsigned mnFrameID;
  
  //Tracking frame id
  long unsigned int mnTrackingFrameID;

  //Used for finding redundant MPs
  long unsigned int mnExamineFuseKF;

  //Local BA
  long unsigned int mnLocalBAKFid;

  //For local BA (Mark fixed KF)
  long unsigned int mnLocalBAFixedKF;

  //Timestemp for KeyFrame
  const double mTimestamp;

  //Define grid for KeyFrame to speed up the Feature matching
  //Similar to Frame class
  const int mnCols;
  const int mnRows;
  //array for Grid
  //define a dynamic  2d vector with type size_t
  vector< vector <vector<size_t> > > mvGrid;
  const float mfGridIndividualWidth;
  const float mfGridIndividualHeight;

  //Camera Parameters
  const float mfx,mfy,mcx,mcy,mInvfx,mInvfy,mbfx,mb,mThreshDepth;

  //Camera calibration Matrix
  const cv::Mat mCalibrationMatrix;

  //KeyPoints
  const int mnNumKP;
  // Vector of keypoints (original) and undistorted keypoints(used by the system).
  // RGB images can be distorted.
  vector<cv::KeyPoint> mvKeyPoints;
  vector<cv::KeyPoint> mvUndisKP;
 //Keypoint corresponds to depth
  vector<float> mvDepthWithKP;
  //Depth INFO
  vector<float> mvDepth;
  //Descriptors for KP
  cv::Mat mDescriptor;
  //BOW INFO
  DBoW2::BowVector mBOWVector;
  DBoW2::FeatureVector mFVector;

  //DBoW3::Vocabulary* mpVocabulary;
  ORBVocabulary* mpVocabulary;

  const int mnScaleLevels;
  const float mfScaleFactor;
  const vector<float> mvScaleFactor;
  //Undistorted Image Bounds.
  float mnMin_X;
  float mnMax_X;
  float mnMin_Y;
  float mnMax_Y;

  //pose and camera centre
  cv::Mat mTWorld2Cam;

  //pose relative to parent
  cv::Mat mTc_father;


  //order
  static bool Order(KeyFrame* pK1, KeyFrame* pK2)
  {
    return pK1->mnID < pK2->mnID;
  }


protected:
  vector<cv::Mat> DescriptorVector(const cv::Mat &Descriptors);


  //-R^(T)*t
  cv::Mat mtCam2World;
  //Translation matrix for transformation from camera to world
  cv::Mat mTCam2World;
  //Centre point for two-camera sensor in world coordination
  cv::Mat mCentreForWorld;

  //The baseline for the sensor
  float mBaseline_Half;

  //Map
  Map* mpMap;
  //relative MP to the KP
  vector<MapPoint*> mvMapPoint;

  //Bad KF
  bool mIsBad;

  map<KeyFrame*, int> mConnectedKFandWeight;//Connected KFs and relative weight
  vector<KeyFrame*> mvpOrderedKFs;//OrderedConnected KFs;
  vector<int> mvOrderedWeights;

  //Tree parameters
  //First time to build the connection, set the Father
  bool mbFirstBuildConnection;
  KeyFrame* mpFather;

  //Children
  set<KeyFrame*> msChildren;


  //lock
  mutex mMutex_Pose;
  mutex mMutex_MP;
  mutex mMutex_Connection;
};

}//end of namespace


#endif // KEYFRAME_H
