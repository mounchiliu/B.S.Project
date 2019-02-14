#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "MapPoint.h"
#include "Map.h"
#include "Frame.h"
#include "KeyFrame.h"

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "mutex"

using namespace std;

namespace ORB_SLAM {

class Optimization
{
public:
  Optimization();
  //Optimize pose only
  void static Optimization_Pose(Frame *pF);
  //local optimization
  void static LocalOptimization(KeyFrame* pKF, Map* pMap, bool* bStop);

};



class Converter
{

public:

  static g2o::SE3Quat Convert2SE3(const cv::Mat &T);

  //Transform to Matrix
  static cv::Mat toCVMatrix(const g2o::SE3Quat &se3);
  static cv::Mat toCVMatrix(const Eigen::Matrix<double,3,1> &em);
  //Transform to Eigen Matrix
  static Eigen::Matrix<double,3,1> toEigenMatrix(const cv::Mat &mat);
  static Eigen::Matrix<double,3,3> toEigenMatrix_3_3(const cv::Mat &mat);

  static vector<float> toQuat(cv::Mat &mat);

};



}



#endif // OPTIMIZATION_H
