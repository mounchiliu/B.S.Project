#ifndef POINTCLOUDMAP_H
#define POINTCLOUDMAP_H

//2nd March PointCloud

#include "KeyFrame.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <pcl/filters/voxel_grid.h>
#include <thread>


namespace ORB_SLAM
{
class PointCloudMap
{
public:

  typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloud;

  PointCloudMap(double resolution);



  //Insert KeyFrame
  void InsertKF(KeyFrame *pKF, cv::Mat& ColorImg,cv::Mat DepthImg);
  void Viewer();
  void ShutDown();

protected:

  PointCloud::Ptr PointCloudGenerator(KeyFrame* pKF, cv::Mat& ColorImg,cv::Mat DepthImg);

  bool mbShutDown;

  //Global Map
  PointCloud::Ptr mPointCloudMap;
  //Viewer(thread)
  shared_ptr<thread> mViewer;

  //Data for drawing
  vector<KeyFrame*> mpKFs;
  vector<cv::Mat> mColorImgs;
  vector<cv::Mat> mDepthImgs;

  //PCL Parameters
  double mResolution;
  pcl::VoxelGrid<pcl::PointXYZRGBA> mVoxel;

  //Lock
  mutex mMutex_KF;
  mutex mMutex_ShutDown;
  mutex mMutex_UpdatingKF;
  condition_variable mUpdatingKF;

  size_t mLastKFsSize;

};



}

#endif // POINTCLOUDMAP_H
