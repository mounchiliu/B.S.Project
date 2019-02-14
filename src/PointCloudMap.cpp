#include "PointCloudMap.h"
#include "Optimization.h"

#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/projection_matrix.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/make_shared.hpp>


namespace ORB_SLAM
{

PointCloudMap::PointCloudMap(double resolution):mbShutDown(false), mLastKFsSize(0)
{
  mResolution = resolution;
  //Downsampling a PointCloud using a VoxelGrid filter
  //setLeafSize (float lx, float ly, float lz)
  //	lx 	the leaf size for X
  //  ly 	the leaf size for Y
  //  lz 	the leaf size for Z
  mVoxel.setLeafSize(mResolution,mResolution,mResolution);
  mPointCloudMap = boost::make_shared<PointCloud>( );

  mViewer = make_shared<thread>( bind(&PointCloudMap::Viewer,this));
}


void PointCloudMap::InsertKF(KeyFrame *pKF, cv::Mat &ColorImg, cv::Mat DepthImg)
{
  //cout<<"PointCloud draws the Keyframe. KeyFrame id = "<<pKF->mnID<<endl;

  unique_lock<mutex> Lock(mMutex_KF);
  mpKFs.push_back(pKF);
  mColorImgs.push_back(ColorImg.clone());
  mDepthImgs.push_back(DepthImg.clone());

  mUpdatingKF.notify_one();
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloudMap::PointCloudGenerator(KeyFrame* pKF, cv::Mat& ColorImg,cv::Mat DepthImg)
{
  PointCloud::Ptr temp(new PointCloud());//nullptr

  //Image INFO
  for(int i = 0;i<DepthImg.rows; i+=3)
  {
    for(int j=0; j<DepthImg.cols; j+=3)
    {
      float d = DepthImg.ptr<float>(i)[j];

      if(d<0.01||d>10)
        continue;

      //World Position
      pcl::PointXYZRGBA point;
      point.z = d;
      point.x = (j-pKF->mcx) * point.z / pKF->mfx;
      point.y = -(i-pKF->mcy) * point.z / pKF->mfy;

      //Color
      point.b = ColorImg.ptr<uchar>(i)[j*3];
      point.g = ColorImg.ptr<uchar>(i)[j*3+1];
      point.r = ColorImg.ptr<uchar>(i)[j*3+2];

      temp->points.push_back(point);

    }
  }

  //Get Camera Pose
  Eigen::Isometry3d T_se3 = Converter::Convert2SE3(pKF->GetPose());
  PointCloud::Ptr cloud_result(new PointCloud);

  // pcl::transformPointCloud (*temp, *result, GlobalTransform);
  pcl::transformPointCloud( *temp, *cloud_result, T_se3.inverse().matrix());
  cloud_result->is_dense = false;

  return cloud_result;
}

void PointCloudMap::Viewer()
{
  pcl::visualization::CloudViewer Viewer("PointCloud Viewer");
  while(1)
  {
    {
      unique_lock<mutex> Lock(mMutex_ShutDown);
      if(mbShutDown)
      {
        break;
      }
    }
    {
      unique_lock<mutex> Lock2(mMutex_UpdatingKF);
      mUpdatingKF.wait(Lock2);
    }

    //KeyFrame Updated
    size_t CurrentKFsSize=0;
    {
      unique_lock<mutex> Lock3(mMutex_KF);
      CurrentKFsSize = mpKFs.size();
    }

    for(size_t i = mLastKFsSize; i<CurrentKFsSize; i++)
    {
      PointCloud::Ptr cloud = PointCloudGenerator(mpKFs[i], mColorImgs[i], mDepthImgs[i]);
      //Add To PointCloud Map
      *mPointCloudMap += *cloud;
    }

    PointCloud::Ptr temp(new PointCloud());
    //filter
    mVoxel.setInputCloud(mPointCloudMap);
    mVoxel.filter(*temp);
    mPointCloudMap->swap(*temp);
    Viewer.showCloud(mPointCloudMap);

    mLastKFsSize = CurrentKFsSize;

  }
}

void PointCloudMap::ShutDown()
{
  {
    unique_lock<mutex> Lock(mMutex_ShutDown);
    mbShutDown = true;
    mUpdatingKF.notify_one();

  }
  mViewer->join();
}

}//END OF NAMESPACE





