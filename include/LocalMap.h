#ifndef LOCALMAP_H
#define LOCALMAP_H

#include "Map.h"
#include "Tracking.h"
#include <mutex>

using namespace std;

namespace ORB_SLAM
{
static double dTime;

class Map;
class Tracking;
class LocalMap
{
public:

  LocalMap(Map* pMap);

  void InsertKFinLocalMap(KeyFrame* pKF);


  //main function
  void run();

  int numKFsWaiting();

  //Set Flags
  void SetAcceptKFs(bool bFlag);
  bool AcceptKFs();
  void BAInterrupt();
  void FinishRequset();
  void SetFinish();
  bool Finished();
  bool CheckFinishQuest();



  //Set Pointer
  void SetTracking(Tracking* pTracking);



protected:


  //Process Inserted KFs
  void ProcessKFs();

  //check whether there has been a KF is in queue
  bool KFsInQueue();

  //Evaluate new added MP
  void EvaluateMPs();

  //Create MPs for new Added MPs (Associate INFO for them)
  void CreateMPs();

  //find redundant MPs in Redundant KFs
  void FindRedundantMPs();

  //Evaluate KeyFrames
  void EvaluateKFs();

  //Compute Fundamental Matrix according to the poses of two KFs
  cv::Mat ComputeFundamentalMatrix(KeyFrame* pKF1, KeyFrame* pKF2);

  cv::Mat FindSkewSymmetricMatrix(cv::Mat &matrix);


  Map* mpMap;

  Tracking* mpTracking;

  list<KeyFrame*> mlNewKFs; // Firstly, it adds KFs from Tracking thread to a waiting list
  list<MapPoint*> mlNewAddedMPs;//For checking
  KeyFrame* mpCurrentKF;


  vector<float> vLocalMap;

  //FLAGS
  bool mbFinish;
  //Not Busy
  bool mbAcceptKFs;

  bool mbRequestFinish;
  bool mbBAinterrupt;



  //Lock
  mutex mMutex_AcceptKFs;
  mutex mMutex_NewKFs;
  mutex mMutex_Finish;



};

}



#endif // LOCALMAP_H
