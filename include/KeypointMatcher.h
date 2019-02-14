#ifndef KEYPOINTMATCHER_H
#define KEYPOINTMATCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "Frame.h"

using namespace std;

namespace ORB_SLAM
{
//Set threshold value
const int HIGHER_THRESHOLD = 100;
const int LOWER_THRESHOLD = 50;
//Number of bins in the counter
const int NumOfBins = 30;

class Frame;

class KeypointMatcher
{

public:
  KeypointMatcher(float ratio=0.6);

  //Use the BOW to speed up the matching
  int MatchUsingBoW(KeyFrame* pKF, Frame &Frame, vector<MapPoint*> &vMatchedMP);

  //Projection between Current Frame and Last Frame
  //This method is used to track from the last frame
  //It returns the number of successful matched pairs
  int ProjectionAndMatch(Frame &Current, Frame &Last, const float threshold);

  //Project each local MP to the current Frame
  int ProjectionAndMatch(Frame &Current, vector<MapPoint*> &mpRefMPs, const float threshold);

  //Triangulation
  int MatchUsingEpipolar(KeyFrame *pKF1, KeyFrame* pKF, cv::Mat &F_2_1, vector<pair<size_t,size_t>> &vMatches);

  //Find redundant MPs in connected KFs
  int FindRedundantMPs(KeyFrame *pKF, vector<MapPoint*> &vpMPs, float r = 3.0f);

  //Count different bits in parallel
  int DescriptorDifference(const cv::Mat &LastDescriptor, const cv::Mat &CurrentDescriptor);



protected:
  float DefineRadiusByCos(float &viewCos);
  void ComputeMajorOrientation(vector<int> *counter,int &max_i1, int &max_i2, int &max_i3);
  bool EpipolarCheck(cv::KeyPoint &kp1,cv::KeyPoint &kp2, cv::Mat &F_2_1, KeyFrame* pKF2);

  //attributes
  float mfRatio;
};

}//END OF namespace

#endif // KEYPOINTMATCHER_H
