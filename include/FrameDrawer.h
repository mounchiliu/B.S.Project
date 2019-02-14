#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <Map.h>
#include <Tracking.h>

using namespace std;

namespace ORB_SLAM{
class Tracking;
class FrameDrawer
{

public:

  //Constructor
  FrameDrawer(Map* pMap);

  //Update info from the last processed frame.
  void Update(Tracking *pTracking);

  //Draw last processed frame.
  cv::Mat DrawFrame();



private:
  void DrawInfo(cv::Mat &image, int state,cv::Mat &imInfo);
  //data member
  //Info of the drawn frame
  cv::Mat mImage;
  //Number of the keypoint
  int mNumKeyPoint;
  int mState;
  vector<cv::KeyPoint> mvCurrentKP;
  int mnTracking;
  int mnTrackedNew;


  Map* mpMap;
  vector<bool> mvMap_MP;
  vector<bool> mvMap_NewMP;

  mutex mMutex;









};

}//END of namespace
#endif // FRAMEDRAWER_H
