#ifndef VIEWER_H
#define VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

namespace ORB_SLAM {

class FrameDrawer;
class MapDrawer;
class Tracking;
class System;

class Viewer
{
public:
  //constructor
  Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking* pTracking, const string &Setting);
  //destructor
  ~Viewer();

  //Draw function
  void run();
  void SetFinishRequest();
  bool Finished();



private:


  //data members
  System* mpSys;
  //Drawers
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;
  Tracking* mpTracking;

  //time = 1/fps in ms
  double mTime;

  float mWidthOfImage, mHeightOfImage;

  //Viwer parameter
  float mViewPointX, mViewPointY, mViewPointZ, mViewPointF;

  //request finish
  bool mbRequestFinish;
  //finished
  bool mbFinish;

  mutex mMutex_ViewerFinish;


  bool FinishCheck();
  void FinishSet();

};


}


#endif // VIEWER_H
