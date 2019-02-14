#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Optimization.h"

namespace ORB_SLAM
{
Tracking::Tracking(FrameDrawer *pFrameDrawer, MapDrawer* pMapDrawer, Map *pMap, shared_ptr<PointCloudMap> pPointCloudMap,
                   const string &strSetting, ORBVocabulary* pVocabulary):mTrackingState(NO_IMAGE), mpFrameDrawer(pFrameDrawer),
                   mpMapDrawer(pMapDrawer), mpMap(pMap), mpViewer(NULL),mpPointCloudMap(pPointCloudMap),
                   mpVocabulary(pVocabulary)
{

  //load camera parameters
  cv::FileStorage fSetting(strSetting, cv::FileStorage::READ);


  float fx = fSetting["Camera.fx"];
  float fy = fSetting["Camera.fy"];
  float cx = fSetting["Camera.cx"];
  float cy = fSetting["Camera.cy"];

  //Calibration Matrix = |fx  0   cx|
  //                     |0   fy  cy|
  //                     |0   0    1|
  // Initialize a diagonal matrix to hold Calibration Matrix
  cv::Mat Calibration = cv::Mat::eye(3,3,CV_32F);
  Calibration.at<float> (0,0) = fx;
  Calibration.at<float> (1,1) = fy;
  Calibration.at<float> (0,2) = cx;
  Calibration.at<float> (1,2) = cy;
  //copy the Matrix to data member for the process of frame latter
  Calibration.copyTo(mCalibration);

  //Undistorted
  //distortion coefficient = [k1,k2,p1,p2,k3]
  cv::Mat DistortCoef(4,1,CV_32F);
  DistortCoef.at<float>(0) = fSetting["Camera.k1"];
  float k1 = DistortCoef.at<float>(0);
  DistortCoef.at<float>(1) = fSetting["Camera.k2"];
  float k2 = DistortCoef.at<float>(1);
  DistortCoef.at<float>(2) = fSetting["Camera.p1"];
  float p1 = DistortCoef.at<float>(2);
  DistortCoef.at<float>(3) = fSetting["Camera.p2"];
  float p2 = DistortCoef.at<float>(3);
  const float k3 = fSetting["Camera.k3"];
  if(k3!=0)
  {
      DistortCoef.resize(5);
      DistortCoef.at<float>(4) = k3;
  }
  DistortCoef.copyTo(mDistort);

  mbfx = fSetting["Camera.bf"];

  float fps = fSetting["Camera.fps"];

  mnMaxKF = fps; //fps-->frame per second

  //Print the Info
  cout << endl << "Camera Calibration Matrix: " << endl;
  cout << " fx: " << fx << endl;
  cout << " fy: " << fy << endl;
  cout << " cx: " << cx << endl;
  cout << " cy: " << cy << endl;
  cout << " k1: " << k1 << endl;
  cout << " k2: " << k2 << endl;
  cout << " k3: " << k3 << endl;
  cout << " p1: " << p1 << endl;
  cout << " p2: " << p2 << endl;
  cout << endl << " fps: " << fps << endl;


  // ORB parameters

  //the number of feature points in each frame
  int n = fSetting["Extractor.nFeatures"];
  //scale for building the image pyramid
  //always set as 1.2
  float scale = fSetting["Extractor.fScaleFactor"];
  //levels of image pyramid
  int levels = fSetting["Extractor.nLevels"];
  //Threshold for FAST detector
  int threshold = fSetting["Extractor.iniThFAST"];
  //if the threshold of detector is not able to get enough keypoints, set the minimum threshold
  int MinThreshold = fSetting["Extractor.minThFAST"];

  mpORBextractor = new ORBextractor(n,scale,levels,threshold,MinThreshold);

  cout << endl  << "ORB Extractor Parameters: " << endl;
  cout << " Number of Features: " << n << endl;
  cout << " Scale Factor: " << scale << endl;
  cout << " Levels in the scale pyramid: " << levels << endl;
  cout << " Fast Threshold: " << threshold << endl;
  cout << " Minimum Fast Threshold: " << MinThreshold << endl;


  mThreshDepth = mbfx*(float)fSetting["ThDepth"]/fx;
  cout << endl << "Depth Threshold (Close/Far Points): " << mThreshDepth << endl;

  mDepth_Map = fSetting["DepthMap"];
  if(fabs(mDepth_Map)<1e-5)
      mDepth_Map=1;
  else
      mDepth_Map = 1.0f/mDepth_Map;

}

cv::Mat Tracking::GrabRGB(const cv::Mat &imageRGB, const cv::Mat &imageDepth, const double &TimeStamp)
{
  mColor = imageRGB;
  mGray = imageRGB;
  mDepth = imageDepth;

  if(mGray.channels() == 3)
  {
    cvtColor(mGray,mGray,CV_RGB2GRAY);
  }


  mDepth.convertTo(mDepth,CV_32F,mDepth_Map);

  mCurrentFrame = Frame(mGray,mDepth,TimeStamp,mpORBextractor,mpVocabulary,mCalibration,mDistort,mbfx,mThreshDepth);

  Track();

  return mCurrentFrame.mTWorld2Cam.clone();

}

void Tracking::Track()
{

  if(mTrackingState == NO_IMAGE)
  {
    mTrackingState = NOT_INITIALIZE;
  }

  mLastState = mTrackingState;

  unique_lock<mutex> lock(mpMap->mMutex_MapUpdate);
  //Initialization
  if(mTrackingState == NOT_INITIALIZE)
  {
    Initialization();

    mpFrameDrawer->Update(this);

    if(mTrackingState != READY)
      return;
  }


  else //Tracking
  {
    bool OperationState;

    if(mTrackingState == READY)
    {

      CheckReplacedMP();//In local Map, the system find redundant MPs and replace MPs

      if(mVelocity.empty())
      {
        OperationState = RefFrameTracking();
      }
      else//Motion Model
      {
        OperationState = MotionModelTracking();
        if(!OperationState)
          OperationState = RefFrameTracking();
      }
    }

    mCurrentFrame.mpReferenceFrame = mpRefFrame;//Set the nearest created KF as Ref KF


    //to get more matches
    if(OperationState)
      OperationState = TrackWithLocalMap();

    if(!OperationState)
      cout<<"Tracking Lost"<<endl;

    if(OperationState)
      mTrackingState = READY;
    else
      mTrackingState = LOST;

    mpFrameDrawer -> Update(this);

    //if tracked
    if(OperationState)
    {
      //Update motion model
      if(!mLastFrame.mTWorld2Cam.empty())
      {
        cv::Mat LastInverseT = cv::Mat::eye(4,4,CV_32F);
        mLastFrame.GetInverseRotation().copyTo(LastInverseT.rowRange(0,3).colRange(0,3));
        mLastFrame.GetCameraCentre().copyTo(LastInverseT.rowRange(0,3).col(3));
        mVelocity = mCurrentFrame.mTWorld2Cam * LastInverseT;
      }
      else
        mVelocity = cv::Mat();


      //Update camera pose for the drawing of Map
      mpMapDrawer->SetCameraPose(mCurrentFrame.mTWorld2Cam);


      //Clear MP created by frame (and do not have matches)
      for(int i=0; i<mCurrentFrame.mNumKeypoints;i++)
      {
        MapPoint* pMP = mCurrentFrame.mvMapPoint[i];
        if(pMP)
          if(pMP->GetNumOfObs()<1)
          {
            mCurrentFrame.mvOutliers[i]=false;
            mCurrentFrame.mvMapPoint[i]=static_cast<MapPoint*>(NULL);
          }

      }


      //delete all temporal MPs
      for(list<MapPoint*>::iterator it = mTemporalMPs.begin(), end = mTemporalMPs.end(); it!=end;it++)
      {
        MapPoint* pMP = *it;
        delete pMP;
      }
      mTemporalMPs.clear();

      if(NeedInsertNewKF())
        InsertNewKF();


      //Delete MPs considered as BAD MPs in bundle adjustment
       for(int i=0;i<mCurrentFrame.mNumKeypoints;i++)
      {
        if(mCurrentFrame.mvMapPoint[i] && mCurrentFrame.mvOutliers[i])
        {
          mCurrentFrame.mvMapPoint[i] = static_cast<MapPoint*>(NULL);
        }
      }

    }

    if(mTrackingState==LOST)
    {
      cout<<"Tracking Lost "<<endl;
    }
    //this is for that not tracked condition
    if(!mCurrentFrame.mpReferenceFrame)
      mCurrentFrame.mpReferenceFrame=mpRefFrame;


    mLastFrame = Frame(mCurrentFrame);
  }


  //Store Pose Info
  //Tracked
  if(!mCurrentFrame.mTWorld2Cam.empty())
  {
    //Record
    //Used to recover camera pose in week tracking
    cv::Mat TCurr_TRefInv=mCurrentFrame.mTWorld2Cam*mCurrentFrame.mpReferenceFrame->GetInversePose();
    mlRecordedCurrRefInv.push_back(TCurr_TRefInv);
    mlRefKFs.push_back(mpRefFrame);//Record for recovery
    mlTimeSample.push_back(mCurrentFrame.mTimeStamp);
  }
  else //Lost
  {
    //Use the last one if tracking lost
    mlRecordedCurrRefInv.push_back(mlRecordedCurrRefInv.back());
    mlRefKFs.push_back((mlRefKFs.back()));
    mlTimeSample.push_back(mlTimeSample.back());
  }
}


void Tracking::Initialization()
{
  //if extract enough KP
  if(mCurrentFrame.mNumKeypoints>500)
  {
    //Set pose to origin
    mCurrentFrame.UpdatePose(cv::Mat::eye(4,4,CV_32F));

    //set the first frame to KF
    //1.build a KF
    KeyFrame* pIniKeyFrame = new KeyFrame(mCurrentFrame,mpMap);

    //2.insert this initial KF in Map
    mpMap->AddKFInMap(pIniKeyFrame);

    cout<<"Current KP for initialization: " << mCurrentFrame.mNumKeypoints << endl;

    //Create relative MP for each feature in initial KF
    for(size_t i = 0; i<mCurrentFrame.mNumKeypoints;i++)
    {

      float depth = mCurrentFrame.mvDepth[i];

      if(depth>0) //if depth of KP available
      {
        //Get three-D point for each feature
        cv::Mat ThreeDPos = mCurrentFrame.BackProject(i);
        //Build MP
        MapPoint* pMP = new MapPoint(ThreeDPos, mpMap, pIniKeyFrame);

        pMP -> AddMPObservation(pIniKeyFrame,i);

        pMP -> ComputeAvgDiscriptor();

        pMP -> UpdateViewDirAndScaleInfo();

        pIniKeyFrame -> AddMapPointToKF(pMP, i);

        mpMap-> AddMPInMap(pMP);

        //Add features as MP for initialisation
        mCurrentFrame.mvMapPoint[i] = pMP;
      }

    }

    cout << "Initialization with  " << mpMap->GetNumOfMP() << " MapPoints. " << endl;

    //Add the KF in local Map
    mpLocalMap->InsertKFinLocalMap(pIniKeyFrame);

    //Set the first frame as REF frame
    mpRefFrame = pIniKeyFrame;
    mCurrentFrame.mpReferenceFrame = pIniKeyFrame;
    //After initialisation, update the last frame using the current frame
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrame_ID = mCurrentFrame.mnId;
    mLastKF = pIniKeyFrame;

    //Store the MP for initialization, //For drawing the Map
    mvRefMapPoints = mpMap->GetMPs();
    //Store KF
    mvRefKeyFrames = mpMap->GetKFs();

    //Set Ref MPs for DRAWING
    mpMap->SetRefMPs(mvRefMapPoints);

    //Update camera pose for the drawing of Map
    mpMapDrawer->SetCameraPose(mCurrentFrame.mTWorld2Cam);

    //Insert KF for PointCloud
    mpPointCloudMap->InsertKF(pIniKeyFrame,mColor,mDepth);

    mTrackingState = READY;
  }
}

bool Tracking::MotionModelTracking()
{
  KeypointMatcher matcher(0.9);

  LastFrameUpdate();

  //set the pose for current frame based on the velocity
  mCurrentFrame.UpdatePose(mVelocity*mLastFrame.mTWorld2Cam);

  //initialize the MP in current frame for matching
  fill(mCurrentFrame.mvMapPoint.begin(), mCurrentFrame.mvMapPoint.end(), static_cast<MapPoint*>(NULL));

  //set searching area threshold
  int threshold = 15;

  //match
  int nMatch = matcher.ProjectionAndMatch(mCurrentFrame,mLastFrame,threshold);
  if(nMatch<20){
    //enlarge the area and match again
    fill(mCurrentFrame.mvMapPoint.begin(), mCurrentFrame.mvMapPoint.end(), static_cast<MapPoint*>(NULL));
    nMatch = matcher.ProjectionAndMatch(mCurrentFrame,mLastFrame,threshold*2);
    if(nMatch<20)
      return false;

  }
  //Optimize camera pose
  Optimization::Optimization_Pose(&mCurrentFrame);

  int nMatch_Map = 0;

  for(int i = 0; i<mCurrentFrame.mNumKeypoints; i++)
  {
    //check MP
    if(mCurrentFrame.mvMapPoint[i])
    {
      if(mCurrentFrame.mvOutliers[i])
      {
        MapPoint* pMp = mCurrentFrame.mvMapPoint[i];
        //discard outliers
        mCurrentFrame.mvMapPoint[i] = static_cast<MapPoint*>(NULL);

        //Update Info
        pMp->mbShouldTrack=false;
        pMp->mnLastSeenFrame = mCurrentFrame.mnId;
        mCurrentFrame.mvOutliers[i] = false;

        nMatch--;
      }
      else if((mCurrentFrame.mvMapPoint[i]->GetNumOfObs())>0)
      {
        nMatch_Map++;
      }
    }

  }

  return nMatch_Map>=10;
}



bool Tracking::RefFrameTracking()
{

  mCurrentFrame.FindBOW();//compute Bow for current frame for latter matching
  KeypointMatcher matcher(0.7f);

  //vector hold matched MP
  vector<MapPoint*> vMatchedMP;

  int nMatch = matcher.MatchUsingBoW(mpRefFrame,mCurrentFrame,vMatchedMP);

  //Do not have enough matches
  if(nMatch<15)
    return false;

  mCurrentFrame.mvMapPoint = vMatchedMP;
  mCurrentFrame.UpdatePose(mLastFrame.mTWorld2Cam);

  //Optimize camera pose
  Optimization::Optimization_Pose(&mCurrentFrame);

  //Discard outliers
  int nMatch_Map = 0;
  for(int i = 0; i<mCurrentFrame.mNumKeypoints; i++)
  {
    //check MP
    if(mCurrentFrame.mvMapPoint[i])
    {
      if(mCurrentFrame.mvOutliers[i])
      {
        MapPoint* pMP = mCurrentFrame.mvMapPoint[i];
        //discard outliers
        mCurrentFrame.mvMapPoint[i] = static_cast<MapPoint*>(NULL);
        //Update Info
        pMP->mbShouldTrack=false;
        pMP->mnLastSeenFrame = mCurrentFrame.mnId;
        mCurrentFrame.mvOutliers[i] = false;

        nMatch--;

      }
      else if(mCurrentFrame.mvMapPoint[i]->GetNumOfObs()>0)
      {
        nMatch_Map++;
      }
    }

  }

  return nMatch_Map>=10;

}

bool Tracking::TrackWithLocalMap()
{
  UpdateLocalMap();
  MatchRefMP();

  //After matching
  mnInlierMatches = 0;
  //Optimize Camera Pose
  Optimization::Optimization_Pose(&mCurrentFrame);


  for(int i = 0; i<mCurrentFrame.mNumKeypoints; i++)
  {
    if(mCurrentFrame.mvMapPoint[i])
    {
      if(!mCurrentFrame.mvOutliers[i])
      {
        mCurrentFrame.mvMapPoint[i]->AddFoundFrame(1);//have founded

        //have match with frames or KFs
        if(mCurrentFrame.mvMapPoint[i]->GetNumOfObs() > 0)
          mnInlierMatches++;
      }
    }
  }

  //strict at the beginning
  if(mCurrentFrame.mnId<mnMaxKF)
    if(mnInlierMatches<50)
      return false;

  if(mnInlierMatches<30)
    return false;
  else
    return true;
}


void Tracking::LastFrameUpdate()
{
  //Get Ref KF of the last frame
  KeyFrame* pRKF = mLastFrame.mpReferenceFrame;
  cv::Mat TRecorded = mlRecordedCurrRefInv.back(); //T_Current * T_Ref.Inv()
  //recover camera pose according to the ref KF when tracking is week
  mLastFrame.UpdatePose(TRecorded*pRKF->GetPose());//Last Pose = _Current * T_Ref.Inv()*T_Ref
                                                   // if the tracking is good, there will be no error

}



void Tracking::UpdateKeyFrame()
{
  //1.Count
  //Record the KF with can observe the MP of the current frame
  //first for KF, second for the number of the KF
  map<KeyFrame*,int> counter;

  for(int i=0; i<mCurrentFrame.mNumKeypoints;i++)
  {
    if( mCurrentFrame.mvMapPoint[i])
    {
      MapPoint* pMP =  mCurrentFrame.mvMapPoint[i];

        if(!pMP->BadMP())
        {
          //For each MP get Obs
          map<KeyFrame*,size_t> Obs = pMP->GetObsInfo();
          //record each info
          for(map<KeyFrame*,size_t>::iterator it = Obs.begin(), end = Obs.end(); it!=end; it++)
          {
            //counter[it->first] to access the second element (use the key to access element) //COUNTER
            //it->first is KF
            //count the MP is observed by which KFs and how many of them
            counter[it->first]++;
          }
        }
        //else if the MP is BAD, delete
        else
        {
          mCurrentFrame.mvMapPoint[i] = NULL;
        }

    }

  }



  if(counter.empty())
  {
    cout<<"In Tracking, no KeyFrame for updating"<<endl;
    return;
  }


  //2. Record each counted KeyFrame to the system
  //clear first
  mvRefKeyFrames.clear();
  mvRefKeyFrames.reserve((3*counter.size()));
  //find the KF can observe the most MPs
  int max = 0;
  KeyFrame* pKF_MAX = static_cast<KeyFrame*>(NULL);
  for(map<KeyFrame*,int>::iterator it = counter.begin(), end = counter.end(); it!=end;it++)
  {
    KeyFrame* pKF = it->first;
    //estimate KF
   if(!pKF->IsBad())
   {
     mvRefKeyFrames.push_back(pKF);
     pKF->mnTrackingFrameID = mCurrentFrame.mnId;

     if(it->second > max)
     {
       pKF_MAX = it->first;
       max = it->second;      
     }
   }
  }
  
  //3. If the Keyframes are limited (not enough), use the Connected KFs
  for(vector<KeyFrame*>::iterator it = mvRefKeyFrames.begin(), end = mvRefKeyFrames.end(); it!= end; it++)
  {
    if(mvRefKeyFrames.size()>80)
      break;

    KeyFrame* pKF = *it;

    //Three methods to find suitable refKFs for tracking

    //1. have most common observation
    vector<KeyFrame*> vConnectedKFs = pKF->GetBestCommonViewKFs(10);//Get connected KFs

    for(vector<KeyFrame*>::iterator itCon=vConnectedKFs.begin(),endCon=vConnectedKFs.end();
        itCon!=endCon; itCon++)
    {
      KeyFrame* pCon = *itCon;
      if(!pCon->IsBad())
      {
        if(pCon->mnTrackingFrameID!=mCurrentFrame.mnId)
        {
          mvRefKeyFrames.push_back(pCon);//Update the connected KF of the this RKF as refKFs in the system for tracking
          pCon->mnTrackingFrameID=mCurrentFrame.mnId;
          break;
        }
      }

    }
    //Use its children
    set<KeyFrame*> sChildren = pKF->GetChildren();//Update children of the connected KF as RefKF in the system
    for(set<KeyFrame*>::iterator it=sChildren.begin(),end=sChildren.end();it!=end;it++)
    {
      KeyFrame* pChild = *it;
      if(!pChild->IsBad())
      {
        if(pChild->mnTrackingFrameID!=mCurrentFrame.mnId)
        {
          pChild->mnTrackingFrameID=mCurrentFrame.mnId;
          mvRefKeyFrames.push_back(pChild);
          break;
        }
      }
    }
    //Search in the parent
    KeyFrame* pFather = pKF->GetFather();
    if(pFather)
      if(pFather->mnTrackingFrameID!=mCurrentFrame.mnId)
      {
        pFather->mnTrackingFrameID=mCurrentFrame.mnId;
        mvRefKeyFrames.push_back(pFather);
        break;
      }

  }
  
  if(pKF_MAX)
  {
    //Update ref frame (The Frame observe most MPs is set as current Ref Frame)
    mpRefFrame = pKF_MAX;
    mCurrentFrame.mpReferenceFrame = pKF_MAX;    
  }
}

void Tracking::UpdateRefMP()
{
  //clear
  mvRefMapPoints.clear();
  
  //For each KF, get MPs as Ref MPs
  for(vector<KeyFrame*>::iterator itKF = mvRefKeyFrames.begin(), endKF = mvRefKeyFrames.end(); itKF!=endKF; itKF++)
  {
    KeyFrame* pKF = *itKF;
    vector<MapPoint*> pMPs = pKF->GetAllMapPoints();
    
    for(vector<MapPoint*>::iterator itMP = pMPs.begin(), endMP = pMPs.end(); itMP!=endMP; itMP++)
    {
      MapPoint* pMP = *itMP;
      if(!pMP)
        continue;
      // avoid repetition
      if(pMP->mnTrackingFrameID == mCurrentFrame.mnId)
        continue;
      if(pMP->BadMP())
        continue;
      mvRefMapPoints.push_back(pMP);
      pMP->mnTrackingFrameID = mCurrentFrame.mnId;
    }
  }  
}

void Tracking::UpdateLocalMap()
{
  UpdateKeyFrame();
  UpdateRefMP();

  //Set Ref MPs for Drawing
  mpMap->SetRefMPs(mvRefMapPoints);




}

//search in the local map --> project the ref MPs to the current frame
void Tracking::MatchRefMP()
{
  //Go through MPs in the Current Frame
  for(vector<MapPoint*>::iterator it = mCurrentFrame.mvMapPoint.begin(), end = mCurrentFrame.mvMapPoint.end(); it != end; it++)
  {
    MapPoint* pMP = *it;
    if(pMP)
    {
      if(pMP->BadMP())
      {
        //delete
        *it = static_cast<MapPoint*>(NULL);
      }
      else//For now the MPs in the current frame is the MP constructed by feature points (In view)
      {
        pMP->AddViewFrame(1);//should be found
        pMP->mnLastSeenFrame = mCurrentFrame.mnId;
        pMP->mbShouldTrack=false;
      }
    }
  }

  //Project all the ref MPs in the current frame
  int NumToMatch = 0;
  for(vector<MapPoint*>::iterator it = mvRefMapPoints.begin(), end = mvRefMapPoints.end(); it != end; it++)
  {
    MapPoint* pMP = *it;

    if(!pMP)
      continue;
    if(pMP->mnLastSeenFrame == mCurrentFrame.mnId)
      continue;
    if(pMP->BadMP())
      continue;


    //Match by Projection
    //Is the MP in the Frustum of the current frame?
    if(mCurrentFrame.MPInViewArea(pMP))
    {
      pMP->AddViewFrame(1);//Num of KF must observe this MP
      NumToMatch++;

    }
  }

  if(NumToMatch>0)
  {
    KeypointMatcher matcher(0.8);
    int threshold = 3;

    //If the camera has been initialize recently, search in a larger area.
    if(mCurrentFrame.mnId<2)
      threshold = 5;

    //Match by projection
    matcher.ProjectionAndMatch(mCurrentFrame,mvRefMapPoints,threshold);

  }
}


void Tracking::CheckReplacedMP()
{
  for(int i=0;i<mLastFrame.mNumKeypoints;i++)
  {
    MapPoint* pMP = mLastFrame.mvMapPoint[i];
    if(pMP)
    {
      MapPoint* pReplaced = pMP->GetReplacedMP();
      if(pReplaced)
        mLastFrame.mvMapPoint[i]=pReplaced;
    }
  }
}

bool Tracking::NeedInsertNewKF()
{ 
  int nKF = mpMap->GetNumOfKF();

  if(mCurrentFrame.mnId<mnMaxKF && nKF>mnMaxKF)
    return false;

  //good tracked points in the ref frame

  int MinObs = 3;
  if(nKF<=2)//few KFs
    MinObs = 2;

  int nRefMatches = mpRefFrame -> GoodTrackedMPs(MinObs);

  //Check if the local map is busy
  bool bLocalMapIdle = mpLocalMap->AcceptKFs();


  //Get the ratio Between tracked close MPs and total close MPs in the Current Frame
  int nNonTrackedCloseMP = 0;
  int nTrackedCloseMP = 0;

  for(int i = 0; i< mCurrentFrame.mNumKeypoints;i++)
  {
    //close points
    if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThreshDepth)
    {
      if(mCurrentFrame.mvMapPoint[i] && !mCurrentFrame.mvOutliers[i])
        nTrackedCloseMP++;
      else
        nNonTrackedCloseMP++;
    }

  }

  bool bInsertClose = (nTrackedCloseMP<100) && (nNonTrackedCloseMP>70);

  //determine threshold
  float RefRatio = 0.75f;
  if(nKF<2)
    RefRatio = 0.4f;


  bool condition1 = mCurrentFrame.mnId >= mnLastKeyFrame_ID+mnMaxKF;
  //week tracking
  bool condition2 = mnInlierMatches<nRefMatches*0.25 || bInsertClose;

  //is local map busy?
  bool condition3 = (mCurrentFrame.mnId>=mnLastKeyFrame_ID && bLocalMapIdle);

  bool condition4 = ((mnInlierMatches<nRefMatches*RefRatio || bInsertClose) && mnInlierMatches>15);

  if((condition1||condition2 || condition3)&&condition4)
  {
    //If the Local Map is idle, insert
    if(bLocalMapIdle)
      return true;
    else
    {
      mpLocalMap->BAInterrupt();

      //Not a lot of KFs in the waiting queue
      if(mpLocalMap->numKFsWaiting()<3)
        return true;
      else
        return false;
    }
  }
  else
    return false;
}


void Tracking::InsertNewKF()
{

  mCurrentFrame.PoseMatrices();
  //construct KF based on the current frame
  KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap);

  //Set it as ref frame
  mpRefFrame = pKF;
  mCurrentFrame.mpReferenceFrame = pKF;

  vector<pair<float,int>> vDepth_Index;
  vDepth_Index.reserve(mCurrentFrame.mNumKeypoints);

  for(int i=0;i<mCurrentFrame.mNumKeypoints;i++)
  {
    if(mCurrentFrame.mvDepth[i]>0)
      vDepth_Index.push_back(make_pair(mCurrentFrame.mvDepth[i],i));
  }

  if(!vDepth_Index.empty())
  {
    //sort and pick the top 100 pairs
    sort(vDepth_Index.begin(),vDepth_Index.end());

    int count = 0;
    for(int j = 0; j<vDepth_Index.size();j++)
    {
      size_t index = vDepth_Index[j].second;
      bool bCreate = false;

      MapPoint* pMp = mCurrentFrame.mvMapPoint[index];
      //Create MP for this curent Frame if the MP is available(Z>0) and has not been created before
      //OR this MP can not been observed by any KF
      if(!pMp)
        bCreate = true;//Create MP for KF using depth
      else if(pMp->GetNumOfObs()<1)
      {
        bCreate = true;
        //clear for later creation
        mCurrentFrame.mvMapPoint[index] = static_cast<MapPoint*>(NULL);
      }

      if(bCreate)
      {
        //Get three-D point for each feature
        cv::Mat ThreeDPos = mCurrentFrame.BackProject(index);
        //Build MP
        MapPoint* pMP_New = new MapPoint(ThreeDPos, mpMap, pKF);

        pMP_New -> AddMPObservation(pKF,index);

        pKF -> AddMapPointToKF(pMP_New, index);

        pMP_New -> ComputeAvgDiscriptor();

        pMP_New -> UpdateViewDirAndScaleInfo();


        mpMap-> AddMPInMap(pMP_New);




        //Store MPs in the corresponding frame
        mCurrentFrame.mvMapPoint[j] = pMP_New;
        count ++;

      }
      else
      {
        count++;
      }

      //Got all MPs whose depth < threshold
      //OR if there are 100 close Map Points, we creat the 100 closest ones;
      if(vDepth_Index[j].first>mThreshDepth && count>100)
        break;
    }
  }

  //Insert KFs to Local Map
  mpLocalMap->InsertKFinLocalMap(pKF);
  mpPointCloudMap->InsertKF(pKF,mColor,mDepth);

  mLastKF = pKF;
  mnLastKeyFrame_ID = mCurrentFrame.mnId;



}

void Tracking::SetLocalMap(LocalMap *pLocalMap)
{
  mpLocalMap=pLocalMap;
}
void Tracking::SetViewer(Viewer *pViewer)
{
  mpViewer = pViewer;
}

}//end of namespace
