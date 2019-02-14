#ifndef SYSTEM_H
#define SYSTEM_H

#include <thread>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "Map.h"
#include "System.h"
#include "BOWVocabulary.h"
#include "Viewer.h"
#include "LocalMap.h"
#include "PointCloudMap.h"

namespace ORB_SLAM
{
class Viewer;
class FrameDrawer;
class Tracking;
class Map;
class LocalMap;
class PointCloudMap;
class System
{

public:
  System(const string &SettinfFile, const string &Vocabulary);

  //Process current rgbd frame
  //Convert RGB to grayscale
  //Return camera pose
  cv::Mat TrackRGB(const cv::Mat image, const cv::Mat &depth, const double &timestamp);


 //shut down systen
  void ShutDownSystem();

  //camera trajectory
  void SaveTrajectory(const string &file);

  //Save keyframe poses
  void SaveKFTrajectory(const string &file);



  //data member

  //Map to store the pointers to keyframes and Mappoints
  Map* mpMap;

  //Tracker to receive a frame and compute the camera pose
  Tracking* mpTracking;

  //Drawers
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;

  //Viewer to show the frame using pangolin
  Viewer* mpViewer;
  std::thread* mptViewer;

  //ORB vocabulary
  ORBVocabulary* mpORBVocabulary;

  //Local Map
  LocalMap* mpLocalMap;
  std::thread* mptLocalMap;


  //PointCloud
  shared_ptr<PointCloudMap> mpPointCloudMap;



};







}


#endif // SYSTEM_H
