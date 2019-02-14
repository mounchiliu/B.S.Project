#include "KeyFrame.h"
namespace ORB_SLAM {

//Initialize ID for previous KeyFrame (mnLastID)
long unsigned int KeyFrame::mnLastID = 0;

KeyFrame::KeyFrame(Frame &Frame, Map *pMap):
  mTimestamp(Frame.mTimeStamp), mnFrameID(Frame.mnId), mnTrackingFrameID(0), mnRows(GRID_COLS), mnCols(GRID_ROWS),
  mfGridIndividualHeight(Frame.mfGridIndividualHeight), mfGridIndividualWidth(Frame.mfGridIndividualWidth),
  mfx(Frame.fx), mfy(Frame.fy), mcx(Frame.cx), mcy(Frame.cy), mInvfx(Frame.invfx), mInvfy(Frame.invfy),
  mbfx(Frame.mbfx), mb(Frame.mb), mThreshDepth(Frame.mThreshDepth), mCalibrationMatrix(Frame.mCalibrationMatrix),
  mnNumKP(Frame.mNumKeypoints), mvKeyPoints(Frame.mvKeyPoints), mvUndisKP(Frame.mvUndisKP),
  mvDepthWithKP(Frame.mvDepthWithKPU), mvDepth(Frame.mvDepth),mDescriptor(Frame.mDescriptors.clone()), mnScaleLevels(Frame.mnScaleLevels),
  mfScaleFactor(Frame.mfScaleFactor), mvScaleFactor(Frame.mvScaleFactors), mBOWVector(Frame.mBOWVector), mFVector(Frame.mFVector),
  mpVocabulary(Frame.mpVocabulary), mnMin_X(Frame.mnMin_X), mnMax_X(Frame.mnMax_X), mnMin_Y(Frame.mnMin_Y), mnMax_Y(Frame.mnMax_Y),
  mBaseline_Half(Frame.mb/2), mpMap(pMap), mvMapPoint(Frame.mvMapPoint), mIsBad(false), mbFirstBuildConnection(true),mpFather(NULL),
  mnExamineFuseKF(0), mnLocalBAKFid(0), mnLocalBAFixedKF(0)
{
  //initialize ID for KeyFrame
  mnID=mnLastID++;
  //initialize vector for Grid
  mvGrid.resize(mnCols);
  for(int i=0;i<mnCols;i++)
  {
    mvGrid[i].resize(mnRows);
    for(int j=0; j<mnRows; j++)
    {
          mvGrid[i][j]=Frame.mvGrid[i][j];
    }
  }

  UpdatePose(Frame.mTWorld2Cam);


}

vector<cv::Mat> KeyFrame::DescriptorVector(const cv::Mat &Descriptor)
{

  vector<cv::Mat> vDescriptor;
  //One row for one descriptor
  vDescriptor.reserve(Descriptor.rows);
  for(int i = 0; i<Descriptor.rows; i++)
  {
    vDescriptor.push_back(Descriptor.row(i));
  }

  return vDescriptor;

}

void KeyFrame::UpdatePose(const cv::Mat &TWorld2Cam)
{
  unique_lock<mutex> Lock(mMutex_Pose); // When updating the pose, can not get it and vice versa

  TWorld2Cam.copyTo(mTWorld2Cam);
  //mTWorld2Cam --> Camera pose (Transfer Matrix)
  cv::Mat RWorld2Cam = mTWorld2Cam.rowRange(0,3).colRange(0,3);
  cv::Mat RCam2World = RWorld2Cam.t();
  cv::Mat tWorld2Cam = mTWorld2Cam.rowRange(0,3).col(3);
  //element in Translation matrix for transformation from camera to world
  //-R^(T)*t
  //The position of the original point of Camera Coordinate in the World coordinate.
  mtCam2World = -RCam2World*tWorld2Cam;

  //initialize
  //four elements for row and col respectively
  mTCam2World = cv::Mat::eye(4,4,TWorld2Cam.type());
  RCam2World.copyTo(mTCam2World.rowRange(0,3).colRange(0,3));
  mtCam2World.copyTo(mTCam2World.rowRange(0,3).col(3));
  //The centre point point for two-camera sensor in world coordination
  cv::Mat centre = (cv::Mat_<float>(4,1) << mBaseline_Half,0,0,1);
  mCentreForWorld=mTCam2World*centre;
}

cv::Mat KeyFrame::GetPose()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mTWorld2Cam.clone();
}

cv::Mat KeyFrame::GetInversePose()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mTCam2World.clone();
}

cv::Mat KeyFrame::CameraCentre()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mtCam2World.clone();
}

cv::Mat KeyFrame::StereoCentre()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mCentreForWorld.clone();
}

cv::Mat KeyFrame::GetRotationMatrix()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mTWorld2Cam.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslationMatrix()
{
  unique_lock<mutex> Lock(mMutex_Pose);
  return mTWorld2Cam.rowRange(0,3).col(3).clone();
}

vector<MapPoint*> KeyFrame::GetAllMapPoints()
{
  unique_lock<mutex> Lock(mMutex_MP);
  return mvMapPoint;
}

MapPoint* KeyFrame::GetMP(size_t &ID)
{
  unique_lock<mutex> Lock(mMutex_MP);
  return mvMapPoint[ID];
}

void KeyFrame::ReplaceMPWith(size_t &ID, MapPoint *pMP)
{
  mvMapPoint[ID]=pMP;
}

void KeyFrame::AddMapPointToKF(MapPoint* pMP, size_t &ID_KP)
{
  unique_lock<mutex> Lock(mMutex_MP);
  mvMapPoint[ID_KP] = pMP;
}

void KeyFrame::DeleteMPinKF(size_t &index)
{
  unique_lock<mutex> Lock(mMutex_MP);
  mvMapPoint[index] = static_cast<MapPoint*>(NULL);//delete MP pointer
}

void KeyFrame::DeleteMPinKF(MapPoint* pMP)
{
  int id = pMP->GetIndexInKF(this);
  if(id>=0)  //Available
    mvMapPoint[id] = static_cast<MapPoint*>(NULL);//delete MP pointer
}

void KeyFrame::FindBOW()
{
  if(mBOWVector.empty() || mFVector.empty())
  {
    //convert the MAT of descriptor to vector
    vector<cv::Mat> vDescriptor = DescriptorVector(mDescriptor);

    //mBOWvector -> (Output) bow vector
    //mFvector -> (Output) feature vector of nodes and feature indexes
    //levelsup ->	levels to go up the vocabulary tree to get the node index
    mpVocabulary->transform(vDescriptor,mBOWVector,mFVector,4);
  }

}


bool KeyFrame::IsBad()
{
  unique_lock<mutex> Lock(mMutex_Connection);
  return mIsBad;
}

void KeyFrame::SetBad()
{
  {
    unique_lock<mutex> Lock(mMutex_Connection);
    if(mnID==0)
      return;
  }
  //Get all Connected KFs
  for(map<KeyFrame*,int>::iterator it = mConnectedKFandWeight.begin(), end=mConnectedKFandWeight.end(); it!=end; it++)
    it->first->EraseConnection(this);
  //erase observation of MPs in this frame (delete the connection between MP and KF)
  for(size_t i=0; i<mvMapPoint.size();i++)
    if(mvMapPoint[i])
      mvMapPoint[i]->EraseMPObservation(this);

  {
    unique_lock<mutex> Lock3(mMutex_Connection);
    unique_lock<mutex> Lock2(mMutex_MP);

    mConnectedKFandWeight.clear();
    mvpOrderedKFs.clear();

    //Update trees
    set<KeyFrame*> sPossibleFatherKF;
    sPossibleFatherKF.insert(mpFather);

    //Update Info for its children
    while(!msChildren.empty())
    {
      bool bChangeFather = false;

      int maxWeight = -1;
      KeyFrame* pChild;
      KeyFrame* pFather;
      //Go through its children KF, update its father
      for(set<KeyFrame*>::iterator itC=msChildren.begin(),endC=msChildren.end();itC!=endC; itC++)
      {
        KeyFrame* pCKF=*itC;
        if(pCKF->IsBad())//Skip bad KFs
          continue;

        //Check the connection between KFs for this child
        vector<KeyFrame*> vpConnectKF = pCKF->GetConnectedKFs();

        //For each connected KF of the child KF, search it in Father KF candidates
        for(size_t i=0; i<vpConnectKF.size(); i++)
        {
          for(set<KeyFrame*>::iterator it=sPossibleFatherKF.begin(),end=sPossibleFatherKF.end();it!=end;it++)
          {
            //if the conneted KF of the child is the Father KF candidates, use it for updating
            if(vpConnectKF[i]->mnID ==(*it)->mnID)
            {
              //Find the max weighted KF of the child's connected KF
              int w = pCKF->GetWeightOfKF(vpConnectKF[i]);//Get the weight of the connected KF of the child
              if(w>maxWeight)
              {
                //Update
                maxWeight = w;
                pChild=pCKF;
                pFather=vpConnectKF[i];
                bChangeFather=true;

              }

           }
          }
        }
      }
      if(bChangeFather)
      {
        pChild->ChangeFather(pFather);
        sPossibleFatherKF.insert(pChild);
        msChildren.erase(pChild);
      }
      else
        break;
    }
    //If there is a child has no connected KF links to possible father KFs, assign original FatherKF to it
    if(!msChildren.empty())
    {
      for(set<KeyFrame*>::iterator it=msChildren.begin(),end=msChildren.end();it!=end;it++)
      {
        (*it)->ChangeFather(mpFather);
      }
    }

    mpFather->DeleteChild(this);
    mTc_father = mTWorld2Cam * mpFather->GetInversePose();
    mIsBad=true;
  }
  mpMap->DeleteKFs(this);
}



int KeyFrame::GoodTrackedMPs(int &minObs)
{
  unique_lock<mutex> Lock(mMutex_MP);

  int count = 0;

  for(int i = 0;i<mnNumKP;i++)
  {
    MapPoint* pMp = mvMapPoint[i];
    if(pMp)
    {
      if(!pMp->BadMP())      
        if(pMp->GetNumOfObs()>=minObs)
          count++;
    }

  }

  return count;
}


void KeyFrame::UpdateConnection()
{
  //Initialize counter
  map<KeyFrame*,int> Counter;
  //1. GET ALL MPs IN KFs
  vector<MapPoint*> vpMPs;
  {
    unique_lock<mutex> Lock(mMutex_MP);
    vpMPs = mvMapPoint;
  }

  //Check in which KFs are they seen
  for(vector<MapPoint*>::iterator it_MP=vpMPs.begin(), end_MP=vpMPs.end(); it_MP!=end_MP; it_MP++)
  {
    MapPoint* pMP = *it_MP;

    if(!pMP)
      continue;
    if(pMP->BadMP())
      continue;

    //Get the observation of MP
    map<KeyFrame*,size_t> Obs = pMP->GetObsInfo();

    for(map<KeyFrame*,size_t>::iterator it_Obs = Obs.begin(), end_Obs=Obs.end(); it_Obs!=end_Obs; it_Obs++)
    {
      //Skip itself, find other KFs
      if((it_Obs->first)->mnID == mnID)
        continue;
      Counter[it_Obs->first]++;
    }
  }

  if(Counter.empty())
    return;

  // 2. find the KF can observe the most MP
  //    Build Connection with the current KF for thoese can observe enough MPs of the current KF
  int max = 0;
  KeyFrame* pKF_max = NULL;
  int threshold = 15;

  vector<pair<KeyFrame*,int>> vPair;
  vPair.reserve(Counter.size());
  for(map<KeyFrame*,int>::iterator it = Counter.begin(), end=Counter.end(); it!=end; it++)
  {
    if(it->second > max)
    {
      pKF_max = it->first;
      max = it->second;
    }
    if(it->second >= threshold)//Over the th, then build connection
    {
      vPair.push_back(make_pair(it->first,it->second));
      (it->first)->BuildConnection(this,it->second);//Build connections with current KF for the KF satisfying the requirement

    }
  }

  //If no one above the th, use the one with max observations.
  if(vPair.empty())
  {
    vPair.push_back(make_pair(pKF_max,max));
    pKF_max->BuildConnection(this,max);//Build connections with the current KF for the KF satisfying the requirement
                                       //This KeyFrame(this) is new added connected KF for the KF satisfying the requirement
  }

  //sort according to the weight(the times seen the MP in the current KF)
  //vPair holds KF with enough common views
  sort(vPair.begin(),vPair.end());
  //Get KFs and Weights
  list<KeyFrame*> KFs;
  list<int> weight;
  for(size_t i=0; i<vPair.size(); i++)
  {
    KFs.push_front(vPair[i].first);
    weight.push_front(vPair[i].second);
  }

  //Get connections between frames
  //Get ordered KFs and Weights
  {
    unique_lock<mutex> Lock2(mMutex_Connection);

    //Update connections
    mConnectedKFandWeight = Counter;
    mvpOrderedKFs = vector<KeyFrame*>(KFs.begin(),KFs.end());
    mvOrderedWeights = vector<int>(weight.begin(),weight.end());


    //Build trees for quick acess
    if(mnID!=0 && mbFirstBuildConnection)
    {
      mbFirstBuildConnection=false;

      //Set the FATHER as the one with the most common views for the first time
      mpFather = mvpOrderedKFs.front();

      //Add child
      mpFather->AddChild(this);
    }
  }
}

//Functions for Building and Updating Connections
void KeyFrame::BuildConnection(KeyFrame *pKF, int &weight)
{
  {
    unique_lock<mutex> Lock(mMutex_Connection);
    if(!mConnectedKFandWeight.count(pKF))//Hasnt added
      mConnectedKFandWeight[pKF]=weight;
    else if(mConnectedKFandWeight[pKF]!=weight)//For updating
      mConnectedKFandWeight[pKF]=weight;
    else
      return;
  }
  UpdateBestCommonViewKFs();
}

void KeyFrame::EraseConnection(KeyFrame *pKF)
{
  bool bUpdate = false;
  {
    unique_lock<mutex> Lock(mMutex_Connection);
    if(mConnectedKFandWeight.count(pKF))
    {
      mConnectedKFandWeight.erase(pKF);
      bUpdate = true;
    }

  }
  if(bUpdate)
    UpdateBestCommonViewKFs();

}

void KeyFrame::UpdateBestCommonViewKFs()
{
  unique_lock<mutex> Lock(mMutex_Connection);

  vector<pair<KeyFrame*,int>> vPair;
  vPair.reserve(mConnectedKFandWeight.size());

  //Currently, mConnectedKFandWeight stores the all connected KF
  for(map<KeyFrame*,int>::iterator it = mConnectedKFandWeight.begin(), end=mConnectedKFandWeight.end(); it!=end; it++)
    vPair.push_back(make_pair(it->first,it->second));

  //sort
  sort(vPair.begin(), vPair.end());
  list<KeyFrame*> KFs;
  list<int> weight;

  for(size_t i=0; i<vPair.size();i++)
  {
    KFs.push_front(vPair[i].first);
    weight.push_front(vPair[i].second);
  }

  //Ordered: Big weight -> small weight
  mvpOrderedKFs = vector<KeyFrame*>(KFs.begin(),KFs.end());
  mvOrderedWeights = vector<int>(weight.begin(),weight.end());
}


vector<KeyFrame*> KeyFrame::GetBestCommonViewKFs(int n)
{
  unique_lock<mutex> Lock(mMutex_Connection);
  if(mvpOrderedKFs.size()<n)
    return mvpOrderedKFs;
  else
    return (vector<KeyFrame*>(mvpOrderedKFs.begin(),mvpOrderedKFs.begin()+n));
}

vector<KeyFrame*> KeyFrame::GetConnectedKFs()
{
  unique_lock<mutex> Lock(mMutex_Connection);
  return mvpOrderedKFs;
}

int KeyFrame::GetWeightOfKF(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Connection);
  if(mConnectedKFandWeight.count(pKF))//Get Weight
    return mConnectedKFandWeight[pKF];
  else
    return 0;
}

//Tree Function
void KeyFrame::AddChild(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Connection);
  msChildren.insert(pKF);
}

void KeyFrame::DeleteChild(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Connection);
  msChildren.erase(pKF);
}

set<KeyFrame*> KeyFrame::GetChildren()
{
  unique_lock<mutex> Lock(mMutex_Connection);
  return msChildren;
}
void KeyFrame::ChangeFather(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Connection);
  mpFather=pKF;
  pKF->AddChild(this);
}
KeyFrame* KeyFrame::GetFather()
{
 unique_lock<mutex> Lock(mMutex_Connection);
 return mpFather;
}

//KeyPoint Functions
vector<size_t> KeyFrame::FeaturesInArea(const float &x, const float  &y, const float  &r) const
{
  //Initialize a vector to store the index of satisfied KeyPoint
  vector<size_t> vIndex;
  vIndex.resize(mnNumKP);

  //mfGridIndividualWidth represents the number of cells for each pixel in a row
  //mfGridIndividualHeight represents the number of cells for each pixel in a col
  //round downward the value
  int nMinCellInX = max(0,(int)floor((x-mnMin_X-r)*mfGridIndividualWidth));  // should be greater than 0
  //shoud not exceed the num of cols defined before
  if(nMinCellInX>=mnCols)
    return vIndex; // the point does not satisfy the requirement

  //round up the value
  //use (mnCols-1) because we need to define a square for searching
  int  nMaxCellInX = min((int)mnCols-1,(int)ceil((x-mnMin_X+r)*mfGridIndividualHeight));
  //should not smaller than 0
  if(nMaxCellInX<0)
    return vIndex;// the point does not satisfy the requirement

  //Similar to the y axis
  //round downward the value
  int nMinCellInY = max(0,(int)floor((y-mnMin_Y-r)*mfGridIndividualWidth));  // should be greater than 0
  //shoud not exceed the num of cols defined before
  if(nMinCellInY>=mnRows)
    return vIndex; // the point does not satisfy the requirement

  //round up the value
  //use (mnRows-1) because we need to define a square for searching
  int  nMaxCellInY = min((int)mnRows-1,(int)ceil((y-mnMin_Y+r)*mfGridIndividualHeight));
  //should not smaller than 0
  if(nMaxCellInY<0)
    return vIndex;// the point does not satisfy the requirement


  for(int temp_x = nMinCellInX; temp_x <= nMaxCellInX; temp_x++)
  {
    for(int temp_y = nMinCellInY; temp_y <= nMaxCellInY; temp_y++)
    {
      //Initialize a vector
      //pass the index of all KPs to this vector
      const vector<size_t> vIndexForKP = mvGrid[temp_x][temp_y];
      if(vIndexForKP.empty()) // If there is no KP at this position
        continue; //skip this condition

      for (size_t i = 0; i < vIndexForKP.size(); i++)
      {
        //Get the undistorted KP of Previous frame
        const cv::KeyPoint &KPU = mvUndisKP[vIndexForKP[i]];
        //x,y -> coordinate of the KeyPoint of last frame projected in current frame
        //this coordinate will be campared with the keypoints in current frame
        float dist_x = fabs(KPU.pt.x - x);
        float dist_y = fabs(KPU.pt.y - y);

        if(dist_x<r && dist_y<r)
        {
          //pass the index of satisfied keypoint
          vIndex.push_back(vIndexForKP[i]);
        }
      }
    }
  }
    return vIndex;
  }


bool KeyFrame::InImage(float &pos_X, float &pos_Y)
{
  return(pos_X >= mnMin_X && pos_X < mnMax_X && pos_Y >= mnMin_Y && pos_Y < mnMax_Y);
}

//Backproject a keypoint into 3D world coordinates
cv::Mat KeyFrame::BackProject(const int &i)
{
  //Get depth
  const float z = mvDepth[i];
  if(z>0)
  {
    const float u = mvUndisKP[i].pt.x;
    const float v = mvUndisKP[i].pt.y;
    const float x = (u-mcx)*z*mInvfx;
    const float y = (v-mcy)*z*mInvfy;
    cv::Mat Three_D =(cv::Mat_<float>(3,1)<<x,y,z);

    unique_lock<mutex> Lock(mMutex_Pose);
    return mTCam2World.rowRange(0,3).colRange(0,3)*Three_D+mTCam2World.rowRange(0,3).col(3);
  }
  else
    return cv::Mat();
}



}//end of namespace
