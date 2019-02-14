#include "LocalMap.h"
#include "Optimization.h"
#include "thread"


namespace ORB_SLAM
{
LocalMap::LocalMap(Map* pMap): mpMap(pMap),mbFinish(true),mbRequestFinish(false),mbBAinterrupt(false),mbAcceptKFs(true)
{
}

//Set the pointer
void LocalMap::SetTracking(Tracking *pTracking)
{
  mpTracking = pTracking;
}

void LocalMap::InsertKFinLocalMap(KeyFrame* pKF)
{
  unique_lock<mutex> Lock(mMutex_NewKFs);
  mlNewKFs.push_back(pKF);
  mbBAinterrupt=true;
}


//Get num of KFs in the waiting queue
int LocalMap::numKFsWaiting()
{
  unique_lock<mutex> Lock(mMutex_NewKFs);
  return mlNewKFs.size();
}

//main function
void LocalMap::run()
{
  mbFinish = false;


  vLocalMap.reserve(1000);



  while(1)
  {
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;

    //Set the thread as busy
    SetAcceptKFs(false);

    //If there is new added KF
    if(KFsInQueue())
    {
      t1 = std::chrono::steady_clock::now();

      //Process New KFs
      ProcessKFs();

      //Evaluate new inserted MP
      EvaluateMPs();


      //Create MPs for new Added MPs (Associate INFO for them)
      CreateMPs();



      //When the sys has processed all the KF
      if(!KFsInQueue())
        //Find redundant MPs
        FindRedundantMPs();


      mbBAinterrupt = false;

      if(!KFsInQueue())
      {
        if(mpMap->GetNumOfKF()>2)
        {


          Optimization::LocalOptimization(mpCurrentKF,mpMap,&mbBAinterrupt);



        }


        EvaluateKFs();




      }


      t2 = std::chrono::steady_clock::now();

      dTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      vLocalMap.push_back(dTime);

    }

    SetAcceptKFs(true);
    if(CheckFinishQuest())
      break;

    //Sleep
    usleep(3000);

  }




  sort(vLocalMap.begin(),vLocalMap.end());

  float totaltime2 = 0;
  int isize = vLocalMap.size();

  for(int ni=0; ni<isize; ni++)
  {
      totaltime2+=vLocalMap[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median Local Mapping time: " << vLocalMap[(isize)/2]<<endl;
  cout << "mean Local Mapping time: " << totaltime2/(isize)<<endl;


  SetFinish();
}

void LocalMap::ProcessKFs()
{
  //1. get one first KF from the waiting list
  {
      unique_lock<mutex> Lock(mMutex_NewKFs);
      mpCurrentKF = mlNewKFs.front();
      mlNewKFs.pop_front();
  }


  mpCurrentKF->FindBOW();//compute Bow for this KF for latter matching

  //2. Get All tracked MPs, associate them with relative KF
  vector<MapPoint*> vpCurrentMPsInKF = mpCurrentKF->GetAllMapPoints();
  for(size_t i = 0;i<vpCurrentMPsInKF.size();i++)
  {
    MapPoint* pMP = vpCurrentMPsInKF[i];
    if(pMP)
    {
      if(!pMP->BadMP())
      {
        //Not in this KF, add INFO
        if(!pMP->inKeyFrame(mpCurrentKF))
        {
          pMP->AddMPObservation(mpCurrentKF,i);
          pMP->UpdateViewDirAndScaleInfo();
          pMP->ComputeAvgDiscriptor();
        }
        else//For new inserted MPs
        {
          //Wait for checking
          mlNewAddedMPs.push_back(pMP);

        }
      }
    }
  }

  //4. Update Links
  mpCurrentKF->UpdateConnection();


  //5. Add KF in Map //In Tracking thread, only the first KF is added to the map directly
                     //Others are passed to the local map thread
  mpMap->AddKFInMap(mpCurrentKF);
}

//check whether there has been a KF is in queue
bool LocalMap::KFsInQueue()
{
  unique_lock<mutex> Lock(mMutex_NewKFs);
  return(!mlNewKFs.empty());
}

//Check new added MPs
void LocalMap::EvaluateMPs()
{
  unsigned long int CurrentKF_ID = mpCurrentKF->mnID;
  list<MapPoint*>::iterator it = mlNewAddedMPs.begin();
  list<MapPoint*>::iterator end=mlNewAddedMPs.end();

  while(it!=end)
  {
    MapPoint* pMP = *it;
    if(pMP->BadMP())
    {
      //Delete from the List
      it = mlNewAddedMPs.erase(it);
    }
    else if(pMP->GetRatioFoundView()<0.25f) // the num of tracked MP should be larger then the predicted num of tracked MPs
    {
      pMP->SetBadMP();
      it = mlNewAddedMPs.erase(it);
    }
    //Few KFs Observe this MP, set as BAD     //  //Set threshold for Obs
                                                  //int Threshold_Obs = 3;
    else if(((int)CurrentKF_ID - (int)pMP->mnFirstObsKF_ID) >= 2 && pMP->GetNumOfObs()<=3)
    {
      pMP->SetBadMP();
      it = mlNewAddedMPs.erase(it);
    }
    else if(((int)CurrentKF_ID-(int)pMP->mnFirstObsKF_ID) >=3)//not in all previous requirements
    {
      //The MP has not been deleted within a long time
      it = mlNewAddedMPs.erase(it); //Erase from the list //Mark it as satisfied MP
    }
    else
      it++;
  }
}

//Create MPs for new Added MPs (Associate INFO for them)
void LocalMap::CreateMPs()
{

  float scaleFactor = 1.5f*mpCurrentKF->mfScaleFactor; //1.5*1.2

  //mathcer
  KeypointMatcher matcher(0.6);

  int numOfKFs = 10;
  vector<KeyFrame*> vpConnectedKFs = mpCurrentKF->GetBestCommonViewKFs(numOfKFs);

  cv::Mat R_w_1 = mpCurrentKF->GetRotationMatrix();
  cv::Mat R_1_w = R_w_1.t();
  cv::Mat t_w_1 = mpCurrentKF->GetTranslationMatrix();
  cv::Mat T_w_1(3,4,CV_32F);
  R_w_1.copyTo(T_w_1.colRange(0,3));
  t_w_1.copyTo(T_w_1.col(3));

  const float &fx_1 = mpCurrentKF->mfx;
  const float &fy_1 = mpCurrentKF->mfy;
  const float &invfx_1 = mpCurrentKF->mInvfx;
  const float &invfy_1 = mpCurrentKF->mInvfy;
  const float &cx_1 = mpCurrentKF->mcx;
  const float &cy_1 = mpCurrentKF->mcy;


  cv::Mat CCInWorld_Current = mpCurrentKF->CameraCentre();

  int numOfNewMP = 0;

  for(size_t i=0; i<vpConnectedKFs.size(); i++)
  {
    if(i>0&&KFsInQueue())
      return;

    KeyFrame* pKF = vpConnectedKFs[i];//pkf -> connected kf

    cv::Mat CCInWorld_Connected = pKF->CameraCentre();

    //Movement of CC(Camera Center)
    cv::Mat move = CCInWorld_Connected - CCInWorld_Current;
    float nMove = cv::norm(move);

    //Movement should be larger than baseline to construct 3D point
    if(nMove<pKF->mb)
      continue;


    //Compute Fundamental Matrix
    cv::Mat F_2_1 = ComputeFundamentalMatrix(mpCurrentKF,pKF);

    //Search matches using epipolar geometry
    vector<pair<size_t,size_t>> vMatches;
    matcher.MatchUsingEpipolar(mpCurrentKF,pKF,F_2_1,vMatches);
    if(vMatches.empty())
      return;

    cv::Mat R_w_2 = pKF->GetRotationMatrix();
    cv::Mat R_2_w = R_w_2.t();
    cv::Mat t_w_2 = pKF->GetTranslationMatrix();
    cv::Mat T_w_2(3,4,CV_32F);
    R_w_2.copyTo(T_w_2.colRange(0,3));
    t_w_2.copyTo(T_w_2.col(3));


    const float &fx_2 = pKF->mfx;
    const float &fy_2 = pKF->mfy;
    const float &invfx_2 = pKF->mInvfx;
    const float &invfy_2 = pKF->mInvfy;
    const float &cx_2 = pKF->mcx;
    const float &cy_2 = pKF->mcy;

    //Triangulate each match
    for(int index=0; index<vMatches.size(); index++)
    {
      //Get the mathced kp
      size_t &id_1 = vMatches[index].first;
      size_t &id_2 = vMatches[index].second;

      cv::KeyPoint &kp1 = mpCurrentKF->mvUndisKP[id_1];
      cv::KeyPoint &kp2 = pKF->mvUndisKP[id_2];
      //Parallax
      bool bRightAvail1 = mpCurrentKF->mvDepthWithKP[id_1] >= 0;
      bool bRightAvail2 = pKF->mvDepthWithKP[id_2] >= 0;

      cv::Mat x1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx_1)*(invfx_1), (kp1.pt.y-cy_1)*invfy_1, 1.0);
      cv::Mat x2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx_2)*(invfx_2), (kp2.pt.y-cy_2)*invfy_2, 1.0);

      //angle of two rays
      cv::Mat Ray_1 = R_1_w * x1;
      cv::Mat Ray_2 = R_2_w * x2;
      //find the angle between two Dirs
      float cosRays = Ray_1.dot(Ray_2)/(cv::norm(Ray_1)*cv::norm(Ray_2));

      float cosParallaxAngle1 = cosRays + 1;//Initialize a value for campare latter
      float cosParallaxAngle2 = cosRays + 1;


      if(bRightAvail1)
        cosParallaxAngle1 = cos(2*atan2(mpCurrentKF->mb/2,mpCurrentKF->mvDepth[id_1]));
      if(bRightAvail2)
        cosParallaxAngle2 = cos(2*atan2(pKF->mb/2,pKF->mvDepth[id_2]));

      float cosParallaxAngle = min(cosParallaxAngle1,cosParallaxAngle2);//find the bigger one

      cv::Mat ThreeDPoint;

      //Parallax angle effective?
      bool bEffective = bRightAvail1 || bRightAvail2 || cosRays < 0.0998;

      if(cosRays<cosParallaxAngle && cosRays >0 && bEffective)//angle of two rays big enough
      {
        //Linear Triangulation Method
        cv::Mat A(4,4,CV_32F);
        A.row(0) = x1.at<float>(0)*T_w_1.row(2)-T_w_1.row(0);
        A.row(1) = x1.at<float>(1)*T_w_1.row(2)-T_w_1.row(1);
        A.row(2) = x2.at<float>(0)*T_w_2.row(2)-T_w_2.row(0);
        A.row(3) = x2.at<float>(1)*T_w_2.row(2)-T_w_2.row(1);

        //Using SVD to find the solution
        //SVD::compute(A, w, u, vt);
        //A->decomposed matrix(i/p)  w->calculated singular values(o/p)
        //u->calculated left singular vectors(o/p) vt->transposed matrix of right singular values
        cv::Mat w,u,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        //sol x is the last column of v  -> last row of vt
        ThreeDPoint = vt.row(3).t();

        //The last element should not be 0
        if(ThreeDPoint.at<float>(3) == 0)
          continue;

        //scale the coordinates to make the last element = 1
        ThreeDPoint = ThreeDPoint.rowRange(0,3)/ThreeDPoint.at<float>(3);
      }//else small angle of two rays will bring large error
      else if(bRightAvail1 && cosParallaxAngle1<cosParallaxAngle2)// depth is available and parallax angle of the first view is bigger
      { //Use back projection to construct the point
        ThreeDPoint = mpCurrentKF->BackProject(id_1);
      }else if(bRightAvail2 && cosParallaxAngle2<cosParallaxAngle1)
      {
        ThreeDPoint = pKF->BackProject(id_2);
      }else
      {
        continue;
      }

      cv::Mat ThreeDPoint_T = ThreeDPoint.t();
      //check triangulation
      //1. whether the point is in front of cameras
      float z1 = R_w_1.row(2).dot(ThreeDPoint_T)+t_w_1.at<float>(2);
      if(z1<=0)
        continue;

      float z2 = R_w_2.row(2).dot(ThreeDPoint_T)+t_w_2.at<float>(2);
      if(z2<=0)
        continue;

      //2. check reprojection error
      float scale_square1 = mpCurrentKF->mvScaleFactor[kp1.octave]*mpCurrentKF->mvScaleFactor[kp1.octave];
      float x_current = R_w_1.row(0).dot(ThreeDPoint_T)+t_w_1.at<float>(0);
      float y_current = R_w_1.row(1).dot(ThreeDPoint_T)+t_w_1.at<float>(1);
      float inv_z1 = 1.0/z1;

      if(!bRightAvail1)
      {
        float u_1 = fx_1*x_current*inv_z1+cx_1;
        float v_1 = fy_1*y_current*inv_z1+cy_1;
        float errorX_1 = u_1 - kp1.pt.x;
        float errorY_1 = v_1 - kp1.pt.y;

        if((errorX_1*errorX_1+errorY_1*errorY_1)>5.991*scale_square1)
          continue;
      }else
      {
        float u_1 = fx_1*x_current*inv_z1+cx_1;
        float v_1 = fy_1*y_current*inv_z1+cy_1;
        float ur_1 = u_1 - mpCurrentKF->mbfx*inv_z1;
        float errorX_1 = u_1 - kp1.pt.x;
        float errorY_1 = v_1 - kp1.pt.y;
        float errorXR_1 = ur_1 - mpCurrentKF->mvDepthWithKP[id_1];
        if((errorX_1*errorX_1+errorY_1*errorY_1+errorXR_1*errorXR_1)>7.8*scale_square1)
            continue;
      }
      //check in the second KF
      float scale_square2 = pKF->mvScaleFactor[kp2.octave]*pKF->mvScaleFactor[kp2.octave];
      float x_2 = R_w_2.row(0).dot(ThreeDPoint_T)+t_w_2.at<float>(0);
      float y_2 = R_w_2.row(1).dot(ThreeDPoint_T)+t_w_2.at<float>(1);
      float inv_z2 = 1.0/z2;

      if(!bRightAvail2)
      {
        float u_2 = fx_2*x_current*inv_z2+cx_2;
        float v_2 = fy_2*x_current*inv_z2+cy_2;
        float errorX_2 = u_2 - kp2.pt.x;
        float errorY_2 = v_2 - kp2.pt.y;

        if((errorX_2*errorX_2+errorY_2*errorY_2)>5.991*scale_square2)
          continue;
      }else
      {
        float u_2 = fx_2*x_2*inv_z2+cx_2;
        float v_2 = fy_2*y_2*inv_z2+cy_2;
        float ur_2 = u_2 - pKF->mbfx*inv_z2;
        float errorX_2 = u_2 - kp2.pt.x;
        float errorY_2 = v_2 - kp2.pt.y;
        float errorXR_2 = ur_2 - pKF->mvDepthWithKP[id_2];
        if((errorX_2*errorX_2+errorY_2*errorY_2+errorXR_2*errorXR_2)>7.8*scale_square2)
            continue;
      }

      // 3. check scale consistency
      //find the distance frome KP to CC1
      cv::Mat Dir1 = ThreeDPoint-CCInWorld_Current;
      cv::Mat Dir2 = ThreeDPoint - CCInWorld_Connected;

      float dist1 = cv::norm(Dir1);
      float dist2 = cv::norm(Dir2);

      if(dist1==0||dist2==0)
        continue;

      float dist_ratio = dist1 / dist2;
      float scales_ratio = mpCurrentKF->mvScaleFactor[kp1.octave] / pKF->mvScaleFactor[kp2.octave];
      //float scaleFactor = 1.5f*mpCurrentKF->mfScaleFactor; //1.5*scalefctor
      // dist1/scale1 * scale2/dist2  > 1.5*1.2 or dist1/scale1 *scale2/dist2 < 1/(1.5*1.2) ->scale consistency
      if(dist_ratio*scaleFactor<scales_ratio || dist_ratio > scaleFactor*scales_ratio)
        continue;

      //Finish checking
      MapPoint* pMP = new MapPoint(ThreeDPoint,mpMap, mpCurrentKF);

      pMP->AddMPObservation(mpCurrentKF, id_1);
      pMP->AddMPObservation(pKF, id_2);
      mpCurrentKF->AddMapPointToKF(pMP, id_1);
      pKF->AddMapPointToKF(pMP, id_2);

      pMP->ComputeAvgDiscriptor();
      pMP->UpdateViewDirAndScaleInfo();

      mpMap->AddMPInMap(pMP);

      mlNewAddedMPs.push_back(pMP);

      numOfNewMP++;

    }

  }

}

void LocalMap::FindRedundantMPs()
{
  //Retrieve connected KFs
  vector<KeyFrame*> vpConnectedKFs = mpCurrentKF->GetBestCommonViewKFs(10);

  vector<KeyFrame*> vpExaminedKFs;
  //For each connected KFs
  for(vector<KeyFrame*>::iterator itKF = vpConnectedKFs.begin(), endKF = vpConnectedKFs.end(); itKF!=endKF; itKF++)
  {
    //Selection
    KeyFrame* pKF = *itKF;
    if(pKF->IsBad() || pKF->mnExamineFuseKF == mpCurrentKF->mnID)
      continue;
    vpExaminedKFs.push_back(pKF);
    //Mark
    pKF->mnExamineFuseKF = mpCurrentKF->mnID;

    //Second-level connected KF
    vector<KeyFrame*> vpSecondConnectedKF = pKF->GetBestCommonViewKFs(5);

    //For each second-level connected KF
    for(vector<KeyFrame*>::iterator itKF2 = vpSecondConnectedKF.begin(), endKF2 = vpSecondConnectedKF.end();
        itKF2!=endKF2; itKF2++)
    {
      KeyFrame* pKF2 = *itKF2;
      if(pKF2->IsBad() || pKF2->mnExamineFuseKF == mpCurrentKF->mnID || pKF2->mnID == mpCurrentKF->mnID)
        continue;
      vpExaminedKFs.push_back(pKF2);
    }

  }


  //Search by reprojection to find fused MPs
  KeypointMatcher matcher;
  //Get all MPs
  vector<MapPoint*> vpMPinCurrentKFs = mpCurrentKF->GetAllMapPoints();

  for(vector<KeyFrame*>::iterator itKF = vpExaminedKFs.begin(), endKF = vpExaminedKFs.end(); itKF!=endKF; itKF++)
  {
    KeyFrame* pKF = *itKF;
    matcher.FindRedundantMPs(pKF,vpMPinCurrentKFs);
  }

  vector<MapPoint*> vpMPCandidates;//used for the selection of MP in KF
  vpMPCandidates.reserve(vpExaminedKFs.size()*vpMPinCurrentKFs.size());

  //Go through each satisfied Connected KFs
  for(vector<KeyFrame*>::iterator itKF=vpExaminedKFs.begin(), endKF=vpExaminedKFs.end(); itKF!=endKF; itKF++)
  {
    KeyFrame* pKF = *itKF;

    //Get all MPs
    vector<MapPoint*> vpMPinKF = pKF->GetAllMapPoints();

    //Go through each MP
    for(vector<MapPoint*>::iterator itMP=vpMPinKF.begin(), endMP=vpMPinKF.end(); itMP!=endMP; itMP++)
    {
      MapPoint* pMP = *itMP;
      if(!pMP)
        continue;
      if(pMP->BadMP())
        continue;
      if(pMP->mnFuseExamineKF==mpCurrentKF->mnID)
        continue;

      //Mark
      pMP->mnFuseExamineKF==mpCurrentKF->mnID;
      vpMPCandidates.push_back(pMP);
    }
  }

  //Project the MPs in connected KFs into the Current KF to find redundant MPs
  matcher.FindRedundantMPs(mpCurrentKF,vpMPCandidates);

  //Update Info
  vpMPinCurrentKFs = mpCurrentKF->GetAllMapPoints();
  for(size_t i = 0;i<vpMPinCurrentKFs.size();i++)
  {
    MapPoint* pMP = vpMPinCurrentKFs[i];
    if(!pMP)
      continue;
    if(pMP->BadMP())
      continue;

    pMP->ComputeAvgDiscriptor();
    pMP->UpdateViewDirAndScaleInfo();
  }
  //Update connections
  mpCurrentKF->UpdateConnection();
}

void LocalMap::EvaluateKFs()
{
  vector<KeyFrame*> vpConnectedKFs = mpCurrentKF->GetConnectedKFs();

  for(vector<KeyFrame*>::iterator it = vpConnectedKFs.begin(),end=vpConnectedKFs.end(); it!=end; it++)
  {
    KeyFrame* pKF = *it;
    if(pKF->mnID==0)//Skip the first frame
      continue;
    vector<MapPoint*> vpMPs = pKF->GetAllMapPoints();

    int nRedundantObs=0;
    int nMPs;

    for(size_t i=0; i<vpMPs.size(); i++)
    {
      MapPoint* pMP = vpMPs[i];
      if(pMP)
      {
        if(!pMP->BadMP())
          //ONLY consider cloase MPs
          if(pKF->mvDepth[i]>pKF->mThreshDepth || pKF->mvDepth[i]<0)
            continue;
          nMPs++;
          if(pMP->GetNumOfObs()>3)//at least have 4 KFs can observe the MP
          {
            int scale = pKF->mvUndisKP[i].octave;
            map<KeyFrame*,size_t> Obs = pMP->GetObsInfo();
            //Count the effective observations
            int num = 0;
            for(map<KeyFrame*,size_t>::iterator itObs=Obs.begin(),endObs=Obs.end(); itObs!=endObs;itObs++)
            {
              KeyFrame* pKF2 = itObs->first;
              if(pKF2==pKF)
                continue;
              int scale2 = pKF2->mvUndisKP[itObs->second].octave;

              if(scale2<=scale+1)
              {
                num++;
                if(num>=3)
                  break;
              }
            }
            if(num>=3)
            {
              nRedundantObs++;
            }
          }
      }

    }
    if(nRedundantObs>0.9*nMPs)
      pKF->SetBad();
  }
}

//Compute Fundamental Matrix according to the poses of two KFs(2->1)
cv::Mat LocalMap::ComputeFundamentalMatrix(KeyFrame* pKF1, KeyFrame* pKF2)
{
  //F = inv(K1)*E*inv(K2)
  //E = t12 ^ R12
  cv::Mat R_w_1 = pKF1->GetRotationMatrix();
  cv::Mat t_w_1 = pKF1->GetTranslationMatrix();
  cv::Mat R_w_2 = pKF2->GetRotationMatrix();
  cv::Mat t_w_2 = pKF2->GetTranslationMatrix();

  cv::Mat R_2_1 = R_w_1*R_w_2.t();
  cv::Mat t_2_1 = -R_w_1*R_w_2.t()*t_w_2 + t_w_1;

  cv::Mat t_2_1_ = FindSkewSymmetricMatrix(t_2_1);

  const cv::Mat &K1 = pKF1->mCalibrationMatrix;
  const  cv::Mat &K2 = pKF2->mCalibrationMatrix;

  return K1.t().inv()*t_2_1_*R_2_1*K2.inv();//3X1(rows,col)
}


cv::Mat LocalMap::FindSkewSymmetricMatrix(cv::Mat &matrix)
{
  return (cv::Mat_<float>(3,3)<<0, -matrix.at<float>(2), matrix.at<float>(1),
      matrix.at<float>(2),0,-matrix.at<float>(0)
      -matrix.at<float>(1), matrix.at<float>(0),0);

}

//Set Flags
void LocalMap::SetAcceptKFs(bool bFlag)
{ 
  unique_lock<mutex> Lock(mMutex_AcceptKFs);
  mbAcceptKFs = bFlag;

}

bool LocalMap::AcceptKFs()
{
  unique_lock<mutex> Lock(mMutex_AcceptKFs);
  return mbAcceptKFs;
}

void LocalMap::FinishRequset()
{
  unique_lock<mutex> Lock(mMutex_Finish);
  mbRequestFinish = true;
}

bool LocalMap::CheckFinishQuest()
{
  unique_lock<mutex> Lock(mMutex_Finish);
  return mbRequestFinish;
}

void LocalMap::SetFinish()
{
  unique_lock<mutex> Lock(mMutex_Finish);
  mbFinish=true;
}

bool LocalMap::Finished()
{
  unique_lock<mutex> Lock(mMutex_Finish);
  return mbFinish;
}

void LocalMap::BAInterrupt()
{
  mbBAinterrupt=true;
}

}
