#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "Map.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "KeypointMatcher.h"

#include <mutex>



#include<opencv2/core/core.hpp>

namespace ORB_SLAM
{
class Map;
class Frame;
class KeyFrame;

class MapPoint
{
public:
  //Initialize MapPoint using frame
  //If the MapPoint is associated with KP in one single frame, this MapPoint will be initialized using Frame
  MapPoint(cv::Mat &Position, Map* pMap, Frame* pFrame, const int &ID_KP);
  //Initialize MapPoint using KeyFrame
  //If the MapPoint is associated with the KeyPoint in several frames, this MapPoint will be initialized using KF
  MapPoint(cv::Mat &Position, Map* pMap, KeyFrame* pKF);

  cv::Mat GetWorldPosition();
  cv::Mat GetViewDir();
  KeyFrame* GetRefFrame();

  void SetWorldPosition(const cv::Mat &Position);

  //Get the map container which stores the INFO about Observation
  std::map<KeyFrame*,size_t> GetObsInfo();
  //Get the number of the camera can observe the MP
  int GetNumOfObs();

  cv::Mat GetDescriptor();

  float GetMaxDist();

  float GetMinDist();

  int GetIndexInKF(KeyFrame* pKF);

  bool BadMP();

  bool SetBadMP();//delete corresponding INFO

  bool inKeyFrame(KeyFrame* pKF);

  //Used to associate the MapPoint with correspongding KeyPoint
  //that is, record the index of the KeyPoint in KeyFrames that can observe this MP
  //For a MP built by KF, it can be observed by several keyframes.
  void AddMPObservation(KeyFrame* pKF,  long unsigned int id_KP);
  //Erase associative INFO about MP and Index of KP
  void EraseMPObservation(KeyFrame* pKF);

  //Replace MP
  void ReplaceWith(MapPoint* pMP);

  //Get replaced MP
  MapPoint* GetReplacedMP();

  //Add num of Frame can view
  void AddViewFrame(int n);

  //Add num of Frame have found  MP
  void AddFoundFrame(int n);

  //Get the Ratio
  float GetRatioFoundView();

  //Compute discriptor for the MP built by KeyFrame
  void ComputeAvgDiscriptor();

  //Compute avg. viewing direction for MP with KFs
  void UpdateViewDirAndScaleInfo();

  //Predict level of the MP
  int PredictLevel(const float &CurrentDist, Frame* pFrame);
  int PredictLevel(float &CurrentDist, KeyFrame* pKF);


  //Attributes

  //INFO for MapPoint
  long unsigned int mnID_MP;//ID for this MapPoint
  static long unsigned int mnLastID;
  long unsigned int mnTrackingFrameID;
  long unsigned int mnLastSeenFrame;
  const long int mnKF_ID;//record the ID of KF for this MP
  const long int mnFrame_ID;//record the ID of frame for this MP;
  const long int mnFirstObsKF_ID;// The ID of the first KF which observes this MP
                                 // For MP constructed by frame, it does not exsit
                                 // For MP constructed by KF, it will be the ID of this KF
  long unsigned int mnFuseExamineKF;//For Finding redundant MPs
  long unsigned int mnLocalBAKFid;//For Local BA
  int mnPredictedLevel;//Predicted level
  bool mbShouldTrack;//Mark the MP should or should not be tracked and projected
  float mfViewCos;//cosine of the angle between the current dir and avg, dir
  float mfProj_X;//Coordinate of the x axis after the projection
  float mfProj_Y;//Coordinate of the y axis after the projection
  float mfProj_XR;//X Coordinate of the Point observed by the right camera and projected to the first camera

  static mutex mMutex;

  //Position in the world coordination
  cv::Mat mWorldPos;

 



private:
  int mnNumObs; // record the number of the camera observed this MP



  //Discriptor for matching
  cv::Mat mDescriptor;


  //Viewing direction for MapPoint
  cv::Mat mViewDir;

  //Num of frames can view
  int mnViewFrame;
  //Num of frame have found MP
  int mnFoundFrame;
  //Ref Frame
  //it should be KeyFrame
  KeyFrame* mpRefFrame;

  //Scale distances
  float mfMinDist;
  float mfMaxDist;

  //record the index of the KeyPoint in KeyFrames that can observe this MP
  //build association
  std::map<KeyFrame*,size_t> mMPObservation;

  //flag to show the MP is deleted
  bool mbBadMP;

  //Replaced by
  MapPoint* mpReplacedWith;

  Map* mpMap;


  mutex mMutex_Pos;
  mutex mMutex_Feature;
};


}// END of namespace

#endif // MAPPOINT_H

