#include "MapPoint.h"

namespace ORB_SLAM
{

long unsigned int MapPoint::mnLastID=0;
mutex MapPoint::mMutex;

//Initialize MapPoint using Frame
//Position --> The world coordinates for this MapPoint
MapPoint::MapPoint(cv::Mat &Position, Map* pMap, Frame* pFrame, const int &ID_KP):
  mnKF_ID(-1), mnFrame_ID(pFrame->mnId), mnTrackingFrameID(0), mnFirstObsKF_ID(-1),mnLastSeenFrame(0), mnNumObs(0),mbBadMP(false),mpRefFrame(static_cast<KeyFrame*>(NULL)),
  mpMap(pMap), mnFoundFrame(1), mnViewFrame(1),mnFuseExamineKF(0), mpReplacedWith((NULL)), mnLocalBAKFid(0)
{
  //Get world coordinates of this point
  Position.copyTo((mWorldPos));
  //get the camera centre in world coordinate
  cv::Mat centre = pFrame->GetCameraCentre();
  //Calculate the viewing direction of this point
  //For Frame, a mappoint is associated with the feature in one single frame
  //The viewing direction is the direction of the vector constructed by the camera centre and the MapPoint
  mViewDir = mWorldPos - centre;
  mViewDir = mViewDir/(cv::norm(mViewDir));

  //Calculate the distance of camera centre and a MapPoint
  double dist = cv::norm(Position-centre);
  //Get the level from which the MapPoint is obtained
  int level = pFrame->mvUndisKP[ID_KP].octave;
  //Get the scale factor of this level
  float ScaleFactor = pFrame->mvScaleFactors[level];
  //Get the levels of the pyramid
  int level_Num = pFrame->mnScaleLevels;

  //                    ____
  //                   /____\     level:n-1 --> dmin  //n for the number of levels
  //                  /______\                       d / 1.2^(n-1-m) = dmin
  //                 /________\   level:m   --> d
  //                /__________\                     dmax / 1.2^m = d
  //Original image /____________\ level:0   --> dmax

  mfMaxDist = dist*ScaleFactor;
  mfMinDist = mfMaxDist / pFrame->mvScaleFactors[level_Num - 1];

  //If the MapPoint is initialized using frame, the discriptor will be the discriptor of this KP
  //mDescriptors in Frame class stores descriptors of all feature points.
  //Each row for each Feature Point
  pFrame->mDescriptors.row(ID_KP).copyTo(mDescriptor);

  unique_lock<mutex> Lock(mpMap->mMutex_CreatePoint);
  mnID_MP=mnLastID++;
}

//Initialize MapPoint using KeyFrame
//Position --> The world coordinates for this MapPoint
MapPoint::MapPoint(cv::Mat &Position, Map* pMap, KeyFrame* pKF):
  mnKF_ID(pKF->mnID), mnFrame_ID(pKF->mnFrameID), mnTrackingFrameID(0), mnFirstObsKF_ID(pKF->mnID),mnLastSeenFrame(0), mnNumObs(0),mbBadMP(false),mpRefFrame(pKF),
  mpMap(pMap), mfMaxDist(0),mfMinDist(0), mnFoundFrame(1), mnViewFrame(1), mnFuseExamineKF(0), mpReplacedWith(static_cast<MapPoint*>(NULL)),
  mnLocalBAKFid(0)
{
  Position.copyTo(mWorldPos);
  //Initialize a null matrix
  mViewDir = cv::Mat::zeros(3,1,CV_32F);
  unique_lock<mutex> Lock(mpMap->mMutex_CreatePoint);
  mnID_MP=mnLastID++;
}

cv::Mat MapPoint::GetWorldPosition()
{
  unique_lock<mutex> Lock(mMutex_Pos);
  return mWorldPos.clone();
}

cv::Mat MapPoint::GetViewDir()
{
  unique_lock<mutex> Lock(mMutex_Pos);
  return mViewDir.clone();
}

KeyFrame* MapPoint::GetRefFrame()
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return mpRefFrame;
}

void MapPoint::SetWorldPosition(const cv::Mat &Position)
{
  unique_lock<mutex> Lock(mMutex_Pos);
  unique_lock<mutex> Lock2(mMutex);
  Position.copyTo(mWorldPos);
}

std::map<KeyFrame*, size_t> MapPoint::GetObsInfo()
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return mMPObservation;
}

int MapPoint::GetNumOfObs()
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return mnNumObs;
}

cv::Mat MapPoint::GetDescriptor()
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return mDescriptor.clone();
}

float MapPoint::GetMaxDist()
{
  unique_lock<mutex> Lock(mMutex_Pos);
  return 1.2f*mfMaxDist;
}

float MapPoint::GetMinDist()
{
  unique_lock<mutex> Lock(mMutex_Pos);
  return 0.8f*mfMinDist;
}

int MapPoint::GetIndexInKF(KeyFrame* pKF)
{
  unique_lock<mutex> lock(mMutex_Feature);
  if(mMPObservation.count(pKF))
    return mMPObservation[pKF];
  else
    return -1;
}

bool MapPoint::BadMP()
{
  unique_lock<mutex> Lock2(mMutex_Feature);
  unique_lock<mutex> Lock(mMutex_Pos);
  return mbBadMP;
}

bool MapPoint::SetBadMP()//Bad Map Point -> delete corresponding INFO
{
  map<KeyFrame*,size_t>Obs;
  {
    unique_lock<mutex> Lock(mMutex_Feature);
    unique_lock<mutex> Lock2(mMutex_Pos);
    mbBadMP = true;
    Obs = mMPObservation;
    mMPObservation.clear();//Clear info
  }
  for(map<KeyFrame*,size_t>::iterator it = Obs.begin(), end=Obs.end(); it!=end; it++)
  {
    KeyFrame* pKF = it->first;
    pKF->DeleteMPinKF(it->second);
  }

  mpMap->DeleteMPs(this);//delete MP in map
}

bool MapPoint::inKeyFrame(KeyFrame* pKF)
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return (mMPObservation.count(pKF));
}

//If the MP can be observed by multiple frames, the MP should be built by KF.
//If the MP is built by KF, the descriptor of this MP is the one which has shortest average dist to others
void MapPoint::ComputeAvgDiscriptor()
{

  vector<cv::Mat> vDescriptor;

  map<KeyFrame*,size_t> Obs;

  {
    unique_lock<mutex> Lock(mMutex_Feature);
    //If the MP has been deleted, return
    if(mbBadMP)
      return;

    Obs = mMPObservation;

  }


  //If has not get any observations, return
  if(Obs.empty())
    return;

  //Initialize the vector holding descriptors
  //The MP initialized by KF can be seen by several Frames
  //The MP can be associated with features in different frames
  //The Num of descriptors for the MP = The NUM of KF
  vDescriptor.reserve(Obs.size());
  //Get all the Descriptors
  for(map<KeyFrame*,size_t>::iterator itr_start = Obs.begin(), itr_end = Obs.end(); itr_start != itr_end; itr_start++)
  {
    //For map container, first for accessing the first element, second for accessing the second element

    //Get the first KF
    KeyFrame* pKF = itr_start->first;

    //If KF is not deleted
    if(!pKF -> IsBad())
    {
      //Add DESCRIPTOR info
      vDescriptor.push_back(pKF->mDescriptor.row(itr_start->second));
    }
  }

  //If no info
  if(vDescriptor.empty())
    return;

  //calculate the difference between each two points
  //num of the descriptor
  size_t num_KP = vDescriptor.size();
  //eg. COL1 holds tall the dists between the first KP and other KPs
  vector<vector<int>> Diff;
  Diff.resize(num_KP,vector<int>(num_KP,0));
  for(size_t i = 0; i < num_KP; i++)
  {
    //Dist to itself
    Diff[i][i] = 0;
    for(size_t j = i+1; j<num_KP; j++)
    {
      //calculate the difference between each two points
      KeypointMatcher* pMatcher;
      int diff = pMatcher->DescriptorDifference(vDescriptor[i],vDescriptor[j]);

      Diff[i][j] = diff;
      Diff[j][i] = diff;

    }
  }

  //Calculate Avg.
  //Calculate the difference between each two point
  //Use the smallest median as the AVG.value
  int BestDiff = 256;
  int Best_ID = 0;
  for(size_t i = 0; i<num_KP; i++)
  {
    //copy
    vector<int> vDif(Diff[i]);
    //sort
    sort(vDif.begin(),vDif.end());

    //get median
    //the num of dif shoud be even
    int middle = vDif[0.5*(num_KP-1)];

    if(middle < BestDiff)
    {
      BestDiff = middle;
      Best_ID = i;
    }
  }

  {
    unique_lock<mutex> Lock2(mMutex_Feature);

    mDescriptor = vDescriptor[Best_ID].clone();
  }


}


void MapPoint::UpdateViewDirAndScaleInfo()
{



  map<KeyFrame*,size_t> Obs;
  KeyFrame* pRef;
  cv::Mat WorldPos;
{
  unique_lock<mutex> lock2(mMutex_Feature);
  unique_lock<mutex> Lock(mMutex_Pos);  

  if(mbBadMP)
    return;

  Obs = mMPObservation;
  pRef = mpRefFrame;
  WorldPos = mWorldPos.clone();

}
  if(Obs.empty())
    return;

  //initialize
  cv::Mat TotalDir = cv::Mat::zeros(3,1,CV_32F);
  int num = 0;
  for(map<KeyFrame*,size_t>::iterator it = Obs.begin(), itend = Obs.end(); it!=itend; it++)
  {
    KeyFrame* pKeyFrame = it->first;

    //get the camera centre in world coordinate
    cv::Mat centreEach = pKeyFrame->CameraCentre();
    //Calculate the viewing direction of this point
    cv::Mat ViewDirEach = mWorldPos - centreEach;
    TotalDir = TotalDir + ViewDirEach/(cv::norm(ViewDirEach));

    num++;
  }



  //Calculate the distance of camera centre and a MapPoint
  cv::Mat dir_ref = WorldPos-pRef->CameraCentre();
  double dist = cv::norm(dir_ref);
  //Get the level from which the MapPoint is obtained
  int level = pRef->mvUndisKP[Obs[pRef]].octave;
  //Get the scale factor of this level
  float ScaleFactor = pRef-> mvScaleFactor[level];
  //Get the levels of the pyramid
  int level_Num = pRef->mnScaleLevels;

  //                    ____
  //                   /____\     level:n-1 --> dmin  //n for the number of levels
  //                  /______\                       d / 1.2^(n-1-m) = dmin
  //                 /________\   level:m   --> d
  //                /__________\                     dmax / 1.2^m = d
  //Original image /____________\ level:0   --> dmax

  {
  unique_lock<mutex> Lock3(mMutex_Pos);

  //Avg.
  mViewDir = TotalDir/num;
  mfMaxDist = dist*ScaleFactor;
  mfMinDist = mfMaxDist / pRef->mvScaleFactor[level_Num - 1];
  }
}


int MapPoint::PredictLevel(const float &CurrentDist, Frame *pFrame)
{
 float ratio;
 {
   unique_lock<mutex> Lock(mMutex_Pos);
   ratio = mfMaxDist/CurrentDist;
 }
 //                    ____
 //                   /____\     level:n-1 --> dmin  //n for the number of levels
 //                  /______\                       d / 1.2^(n-1-m) = dmin
 //                 /________\   level:m   --> d
 //                /__________\                     dmax / 1.2^m = d       --> 1.2^m = dmax/d  --> m = log(dmax/d)/log(1.2)
 //Original image /____________\ level:0   --> dmax


 float LogScaleFactor = log(pFrame->mfScaleFactor);
 int nLevel = ceil(log(ratio)/LogScaleFactor);

 if(nLevel<0)
  nLevel = 0;
 else if(nLevel >= pFrame->mnScaleLevels)
 {
   nLevel = pFrame->mnScaleLevels - 1;
 }

 return nLevel;
}

int MapPoint::PredictLevel(float &CurrentDist, KeyFrame *pKF)
{
 float ratio;
 {
   unique_lock<mutex> Lock(mMutex_Pos);
   ratio = mfMaxDist/CurrentDist;
 }
 //                    ____
 //                   /____\     level:n-1 --> dmin  //n for the number of levels
 //                  /______\                       d / 1.2^(n-1-m) = dmin
 //                 /________\   level:m   --> d
 //                /__________\                     dmax / 1.2^m = d       --> 1.2^m = dmax/d  --> m = log(dmax/d)/log(1.2)
 //Original image /____________\ level:0   --> dmax


 float LogScaleFactor = log(pKF->mfScaleFactor);
 int nLevel = ceil(log(ratio)/LogScaleFactor);

 if(nLevel<0)
  nLevel = 0;
 else if(nLevel >= pKF->mnScaleLevels)
 {
   nLevel = pKF->mnScaleLevels - 1;
 }

 return nLevel;
}




void MapPoint::AddMPObservation(KeyFrame* pKF,  long unsigned int id_KP)
{
  unique_lock<mutex> Lock(mMutex_Feature);
  //If the pair of KP index and KF is added in the map container
  if(mMPObservation.count(pKF))
    return; //return
  //Otherwise
  //add association
  //Passing the index of KP id_KP into the Map
  mMPObservation[pKF] = id_KP;

  if(pKF->mvDepthWithKP[id_KP]>=0)
    mnNumObs += 2;
  else
    mnNumObs++;//for mps cannot be observed by depth camera

}



void MapPoint::EraseMPObservation(KeyFrame *pKF)
{
  bool bBadMP = false;
  {
  unique_lock<mutex> Lock(mMutex_Feature);
  //If the pair of KP index and KF is added in the map container
  if(mMPObservation.count(pKF))
  {
    //Update the num of cameras observing the MP(KP)
    if(pKF->mvDepthWithKP[mMPObservation.count(pKF)])
      mnNumObs -= 2;
    else
      mnNumObs -=1;

    //Erase the INFO
    mMPObservation.erase(pKF);

    //If this KF is the ref frame
    if(pKF==mpRefFrame)
      //re-defined the ref KF
      mpRefFrame = mMPObservation.begin()->first;

    //After Updating the INFO, check the number of cameras observing this MP
    //If the num of cameras observing the MP is less than 2 which is illega
    //PS: if it equals, it means that the MP can only obsered in one frame --> MP should built by Frame
    //delete the MP
    if(mnNumObs<=2)
      bBadMP = true;
  }
  }
  if(bBadMP)
    SetBadMP();
        //Let corresponding Frame know that the MP is deleted
}


void MapPoint::AddViewFrame(int n)
{
  unique_lock<mutex> Lock(mMutex_Feature);
  mnViewFrame+=n;
}

void MapPoint::AddFoundFrame(int n)
{
  unique_lock<mutex> Lock(mMutex_Feature);
  mnFoundFrame+=n;
}

float MapPoint::GetRatioFoundView()
{
  unique_lock<mutex> Lock(mMutex_Feature);
  return static_cast<float>(mnFoundFrame)/mnViewFrame;
}

void MapPoint::ReplaceWith(MapPoint *pMP)
{
  //Avoid to erase the same MP
  if(pMP->mnID_MP == this->mnID_MP)
    return;

  int nView, nFound;
  //For loser MP, get all the KFs can observe it
  map<KeyFrame*,size_t>Obs;
  {
    unique_lock<mutex> Lock(mMutex_Feature);
    unique_lock<mutex> Lock2(mMutex_Pos);
    mbBadMP = true;
    Obs = mMPObservation;
    mMPObservation.clear();//Clear info
    nView = mnViewFrame;
    nFound = mnFoundFrame;
    mpReplacedWith = pMP;
  }

  //Replace the MP in all the KF which can observe the MP
  for(map<KeyFrame*,size_t>::iterator it=Obs.begin(), end=Obs.end(); it!=end; it++)
  {
    KeyFrame *pKF = it->first;

    if(!pMP->inKeyFrame(pKF))//MP not in the Obs of this MP(has not added in the KF), replace
    {
      pKF->ReplaceMPWith(it->second, pMP);//Use the pMP to replace the MP in KF
      pMP->AddMPObservation(pKF,it->second);//Add Observation
    }
    else//Erase
    {
      //Remove redundant MP
      pKF->DeleteMPinKF(it->second);
    }

  }

  pMP->AddFoundFrame(nFound);
  pMP->AddViewFrame(nView);
  //pMP->ComputeAvgDiscriptor();

  mpMap->DeleteMPs(this);

}

MapPoint* MapPoint::GetReplacedMP()
{
  unique_lock<mutex>  Lock(mMutex_Feature);
  unique_lock<mutex> Lock2(mMutex_Pos);
  return mpReplacedWith;

}

}//END of namespace

