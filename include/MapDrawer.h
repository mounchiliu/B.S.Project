#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "Map.h"
#include <pangolin/pangolin.h>


namespace ORB_SLAM {

class MapDrawer
{
public:
  //Initialize
  MapDrawer(Map* pMap, const string &SettingFile);
  //Set the camera pose
  void SetCameraPose(cv::Mat &CurrentPose);
  //Get current camera matrix (output is in the form of opengl Matrix)
  void GetCUrrentCameraPose(pangolin::OpenGlMatrix &Output_CameraMatrix);
  //Draw the camera
  void DrawCurrentCamera(pangolin::OpenGlMatrix &CameraPose);
  //Draw Map Points
  void DrawMapPoint();
  //Draw Camera Trajectory
  void DrawKeyFrame();


private:
  Map* mpMap;
  cv::Mat mCameraPose;
  float mSizeOfPoint;
  //The radio between width of the camera and the total width
  float mSizeOfCamera;

  //LOCK
  mutex mMutex_Camera;
};
}


#endif // MAPDRAWER_H
