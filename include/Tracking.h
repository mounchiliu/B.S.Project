#ifndef TRACKING_H
#define TRACKING_H

#include "Frame.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "ORBextractor.h"
#include "Map.h"
#include "KeyFrame.h"
#include "BOWVocabulary.h"
#include "PointCloudMap.h"
#include "LocalMap.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <math.h>
#include <mutex>
using namespace std;

namespace ORB_SLAM
{
class Map;
class Viewer;
class FrameDrawer;
class System;
class PointCloudMap;
class LocalMap;

class Tracking
{

public:
  //function
  Tracking(FrameDrawer *pFrameDrawer, MapDrawer* pMapDrawer, Map *pMap, shared_ptr<PointCloudMap> pPointCloudMap, const string &strSetting, ORBVocabulary* pVocabulary);
  cv::Mat GrabRGB(const cv::Mat &imageRGB, const cv::Mat &imageDepth, const double &TimeStamp);

  //Set Pointers
  void SetLocalMap(LocalMap* pLocalMap);

  void SetViewer(Viewer* pViewer);

  //data member
  enum eState{
    NOT_READY = -1,
    NO_IMAGE = 0,
    NOT_INITIALIZE = 1,
    READY = 2,
    LOST = 3
  };

  //Current state
  eState mTrackingState;
  //Last state
  eState mLastState;

  //Current Frame
  Frame mCurrentFrame;
  //Imgs for current Frame
  //color image for drawing point cloud
  cv::Mat mColor;
  //Grayscale image
  cv::Mat mGray;
  //Depth Imgs
  cv::Mat mDepth;

  //Recorded Pose For Recovery
  list<cv::Mat> mlRecordedCurrRefInv;
  //Recorded RefKF
  list<KeyFrame*> mlRefKFs;
  list<double> mlTimeSample;







protected:

  void Track();

  //Initialization
  void Initialization();

  //Track with motion model
  bool MotionModelTracking();

  //Track with Reference Frame
  bool RefFrameTracking();

  //Track with local Map
  bool TrackWithLocalMap();

  //Update Last Frame to build new MP
  void LastFrameUpdate();


  //Record the keyframes with can observe the MPs of current frame
  void UpdateKeyFrame();
  
  //Update MP for pose estimation
  void UpdateRefMP();
  
  //Update local map info
  void UpdateLocalMap();

  //Check Replaced MP
  void CheckReplacedMP();
  
  //Match Reference MPs
  void MatchRefMP();

  //Estimate whether we need insert a KF
  bool NeedInsertNewKF();

  //Insert new KF
  void InsertNewKF();

  //Track with motion mode
  //For ORB extractor
  ORBextractor* mpORBextractor;

  //Vocabulary
  ORBVocabulary* mpVocabulary;

  //Calibration Matrix
  cv::Mat mCalibration;
  //Distortion
  cv::Mat mDistort;
  //Baseline multiplied by fx.
  float mbfx;
  //Close/Far Threshold
  //Close points seen by the sensor are reliable
  //CLose points are inserted from just one frame
  //For Far points, they require matches in two frames
  float mThreshDepth;
  //depth<->Map FACTOR
  float mDepth_Map;

  int mnMaxKF;


  //last Ref Frame
  KeyFrame* mpRefFrame;

  //Drawers
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;

  //Map
  Map* mpMap;

  //ref MP for pose estimation
  vector<MapPoint*> mvRefMapPoints;
  //refKF
  vector<KeyFrame*> mvRefKeyFrames;


  //For Motion Model
  cv::Mat mVelocity;

  //Current matches
  int mnInlierMatches;

  //Last KF, Frame, Relocalization Frame
  //The Last KF
  KeyFrame* mLastKF;
  //Last Frame
  Frame mLastFrame;
  unsigned int mnLastKeyFrame_ID;

  //temporal MPs
  list<MapPoint*> mTemporalMPs;

  LocalMap* mpLocalMap;
  Viewer* mpViewer;

  //PointCloud
  shared_ptr<PointCloudMap> mpPointCloudMap;

};




}//end of namespace
#endif // TRACKING_H
