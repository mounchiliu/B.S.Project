#include "KeypointMatcher.h"

#include<stdint-gcc.h>


namespace ORB_SLAM
{

KeypointMatcher::KeypointMatcher(float ratio): mfRatio(ratio)
{
}

int KeypointMatcher::ProjectionAndMatch(Frame &Current, Frame &Last, const float threshold)
{
  int nNumMatches = 0;

  //Estimate the matching to remove error
  //collect the orientation of the pair of KeyPoints
  //NumOfBins = 30;
  vector<int> counter[NumOfBins];
  //The counter counts the angle differnce between the Last KP and the best mathced current KP
  //initialize memory for each single bin
  for(int i=0; i<NumOfBins; i++)
  {
    counter[i].reserve(500);
  }

  //Get the coordinates of the camera centre of current Frame in world coordination
  const cv::Mat Rworld2cam = Current.mTWorld2Cam.rowRange(0,3).colRange(0,3);
  const cv::Mat tworld2cam = Current.mTWorld2Cam.rowRange(0,3).col(3);

  const cv::Mat CentreInWorld = -Rworld2cam.t()*tworld2cam;

  //Get the coordinates of the camera centre of current Frame in Last Frame
  const cv::Mat Rworld2cam_Last = Last.mTWorld2Cam.rowRange(0,3).colRange(0,3);
  const cv::Mat tworld2cam_Last = Last.mTWorld2Cam.rowRange(0,3).col(3);

  //With the rotation and translation, the point of current camera centre can be projected into current Frame
  const cv::Mat CurrentCentreInLast = Rworld2cam_Last*CentreInWorld+tworld2cam_Last;



  //Estimate forward or backwoard
  bool bMoveForward = CurrentCentreInLast.at<float>(2) > Current.mb;
  bool bMoveBackward = -CurrentCentreInLast.at<float>(2) > Current.mb;

  for(int i=0; i<Last.mNumKeypoints; i++)
  {
    //Get the MP corresponds to the KP in Last Frame
    MapPoint* pMP_Last = Last.mvMapPoint[i];

    //If pMP has a value
    if(pMP_Last)
    {
      if(!Last.mvOutliers[i])//if the MP is not outer the area
      {
        cv::Mat MPinWorld = pMP_Last->GetWorldPosition();
        //Get the position of the MP of the Last Frame in the Current Frame
        cv::Mat LastMPInCurrent = Rworld2cam*MPinWorld+tworld2cam;

        float x_LastInCurrent = LastMPInCurrent.at<float>(0);
        float y_LastInCurrent = LastMPInCurrent.at<float>(1);
        float invz_LastInCurrent = 1.0/LastMPInCurrent.at<float>(2);

        //check the depth INFO
        if(invz_LastInCurrent < 0)
          continue;

        //Get the pixel coordinate of KP(MP) of the Last Frame in Current Frame
        float u_Last = Current.fx*x_LastInCurrent*invz_LastInCurrent+Current.cx;
        float v_Last = Current.fy*y_LastInCurrent*invz_LastInCurrent+Current.cy;

        //Is it within the bound
        if(u_Last < Current.mnMin_X || u_Last > Current.mnMax_X || v_Last < Current.mnMin_Y || v_Last > Current.mnMax_Y)
          continue;

        //Get the level of pyramid for this KP in Last Frame
        int nLastLevel = Last.mvKeyPoints[i].octave;

        //Define a searching area
        //Corresponds to the scale for each level of pyramid
        float r = threshold*Current.mvScaleFactors[nLastLevel];

        //Initialize a vector to store the indices of satisfied KP
        vector<size_t> vIndex;

        //the Current Frame moves farward, the image for the Current Frame is larger
        //to get lower size image which corresponds to the image size of the Last Frame
        //should search in the levels bigger than nLastLevel to search in smaller images
        if(bMoveForward)
        {
          vIndex = Current.FeaturesInArea(u_Last,v_Last,r,nLastLevel);
        }
        //the Current Frame moves backward, the image for the Current Frame is smaller
        //to match with image with larger size --> search in level smaller than nLastLevel
        else if(bMoveBackward)
        {
          vIndex = Current.FeaturesInArea(u_Last,v_Last,r,0,nLastLevel);
        }
        else //the same distance to the OBJ
        {
          vIndex = Current.FeaturesInArea(u_Last,v_Last,r,nLastLevel-1,nLastLevel+1);
        }

        //if does not get any matches
        if(vIndex.empty())
          continue; //skip this condition

        //Totally different
        int nBestDistance = 256;
        //Initialize ID
        int nBestMachedKP_ID = -1;

        //Get the descriptor of this MP in the Last Frame
        const cv::Mat MP_Last_Descriptor = pMP_Last->GetDescriptor();


        //go through every KP obtained before
        for(vector<size_t>::iterator vbegin = vIndex.begin(),vend = vIndex.end(); vbegin!=vend; vbegin++)
        {

          size_t index = *vbegin;         

          if(Current.mvMapPoint[index])
            if(Current.mvMapPoint[index]->GetNumOfObs() > 0)
              continue;

          //the x value of same point in right camera projected into the first camera
          if(Current.mvDepthWithKPU[index]>0)
          {
            //GET The x value of the point in right camera projected into the left camera
            float x2 = u_Last - Current.mbfx*invz_LastInCurrent;
            //Ensure the point in the right camera is also within area determined by radius r
            //Get the difference between the x values of the two points in the right camera
            //projected in the left camera
            //One for the point in the Last Frame projected into the Current Frame
            //One for the point in the Current Frame
            float errorOfRight = fabs(x2-Current.mvDepthWithKPU[index]);

            //If beyond r
            if(errorOfRight>r)
              continue;//skip
          }

          //get the descriptor of matched KP in the Current Frame
          const cv::Mat &Descriptor_Current = Current.mDescriptors.row(index);

          //Calculate the distance(difference)
          int ndist = DescriptorDifference(MP_Last_Descriptor,Descriptor_Current);

          //Replace if satisfied
          if(ndist<nBestDistance)
          {
            nBestDistance = ndist;
            nBestMachedKP_ID = index;
          }
        }

        //if below the DETERMINED THRESHOLD
        if(nBestDistance<=HIGHER_THRESHOLD)
        {
          //Add MP through searching
          //Add the matched MP in Last Frame to the Current Frame (The same MP shares the same position in world coordinate
          //Build association between the Current Frame and the MP in the Last Frame
          Current.mvMapPoint[nBestMachedKP_ID] = pMP_Last;
          //Update variable for the NUM of matches
          nNumMatches++;


          //For each matched point, collect the orientation of the pair of KeyPoints
          float diff_angle = Last.mvUndisKP[i].angle - Current.mvUndisKP[nBestMachedKP_ID].angle;
          if(diff_angle<0)
          {
            //adjust the angle diff
            diff_angle = diff_angle+360.0f;
          }

          //calculate which group the point should be
          int IndexForBin = round(diff_angle / (1.0f / NumOfBins));
          if(IndexForBin==NumOfBins)
            IndexForBin = 0;
          //put it into the counter
          if(IndexForBin < NumOfBins && IndexForBin >=0)
          {
            //i is for the index of the counter
            //in each index, it contains the ID of matched KP
            counter[IndexForBin].push_back(nBestMachedKP_ID);
          }
        }
      }
    }
  }



  //The orientation of the image must match the orientation of the KP

  int i1 = -1; int i2 = -1; int i3 = -1;
  //find three orientation with the maximum counted value (Major orientation)
  ComputeMajorOrientation(counter,i1,i2,i3);

  //if the difference angle of the matched pair is not within the three types of orientation, delete
  for(int i = 0; i<NumOfBins; i++)
  {
    if(i!=i1 && i!=i2 && i!=i3)
    {
      for(int j = 0; j<counter[i].size(); j++)
      {
        //acess the assigned KP to delete
        //counter[i][j] to acess the ID of matched KP, i.e. the KP is not consistent with the major orientation
        Current.mvMapPoint[counter[i][j]] = static_cast<MapPoint*>(NULL);
        nNumMatches--;
      }
    }
  }

  return nNumMatches;
}

//In the function in the Tracking.cpp, we have use the method in the frame class to project the MP to the current frame
//to record whether it is in the frustum
int KeypointMatcher::ProjectionAndMatch(Frame &Current, vector<MapPoint*> &mpRefMPs, const float threshold)
{
  int nNumMatch = 0;
  //go through all the RefMps
  for(size_t i= 0; i<mpRefMPs.size();i++)
  {
    MapPoint* pMP = mpRefMPs[i];

    if(pMP->mbShouldTrack==false)
      continue;
    if(pMP->BadMP())
      continue;

    //define the level of the MP and the searching area
    int &nPredictedLevel = pMP->mnPredictedLevel;
    float r = DefineRadiusByCos(pMP->mfViewCos);
    r = r*threshold*Current.mvScaleFactors[nPredictedLevel];

    //find the relative feature point according to the MP
    vector<size_t> vIndex = Current.FeaturesInArea(pMP->mfProj_X,pMP->mfProj_Y,r,nPredictedLevel-1,nPredictedLevel);

    if(vIndex.empty())
    {
      continue;
    }

    //get descriptor
    //Calculate dff
    const cv::Mat Descriptor_MP = pMP->GetDescriptor();

    //Find the two best matched KP
    int bestdist = 256;
    int bestID = -1;
    int bestdist2 = 256;

    int levelforBestdist = -1;
    int levelforBestdist2 = -1;

    //go through each feature point
    for(vector<size_t>::iterator it=vIndex.begin(),end=vIndex.end();it!=end;it++)
    {
      //get id
      size_t id = *it;

      //if the feature point is matched
      if(Current.mvMapPoint[id])
        if(Current.mvMapPoint[id]->GetNumOfObs() > 0)
          continue;

      if(Current.mvDepthWithKPU[id]>0)
      {
        //Ensure the point in the right camera is also within area determined by radius r
        //Get the difference between the x values of the two points in the right camera
        //projected in the left camera
        //One for the point in the Last Frame projected into the Current Frame
        //One for the point in the Current Frame
        float errorOfRight = fabs(pMP->mfProj_XR-Current.mvDepthWithKPU[id]);

        //If beyond r
        if(errorOfRight>r)
          continue;//skip
      }

      const cv::Mat &descriptor_KP = Current.mDescriptors.row(id);

      //Calculate the diff
      const int dist = DescriptorDifference(Descriptor_MP,descriptor_KP);

      //find the best two ones
      if(dist<bestdist)
      {
        bestdist2 = bestdist;
        levelforBestdist2 = levelforBestdist;
        bestdist = dist;
        levelforBestdist = Current.mvUndisKP[id].octave;
        bestID = id;

      }else if(dist<bestdist2)
      {
        bestdist2 = dist;
        levelforBestdist2 = Current.mvUndisKP[id].octave;
      }
    }

    if(bestdist<=HIGHER_THRESHOLD)
    {
      //in the same level and the bestDist is not similar enough, SKIP
      if(levelforBestdist == levelforBestdist2 && bestdist > mfRatio*bestdist2)
        continue;

      //In different level use the best one
      //In the same level and dist is small enough, use the best one
      Current.mvMapPoint[bestID] = pMP;
      nNumMatch++;
    }
  }

  return nNumMatch;
}

int KeypointMatcher::MatchUsingEpipolar(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat &F_2_1, vector<pair<size_t,size_t>> &vMatches)
{

  DBoW2::FeatureVector &vFeature_Vec1 = pKF1->mFVector;
  DBoW2::FeatureVector &vFeature_Vec2 = pKF2->mFVector;

  cv::Mat CameraCenter = pKF1->CameraCentre();//CC of KF1 in world
  cv::Mat R_w_2 = pKF2->GetRotationMatrix();
  cv::Mat t_w_2 = pKF2->GetTranslationMatrix();
  cv::Mat CameraCenter2 = R_w_2*CameraCenter+t_w_2; //CC of KF1 in Camera 2

  float inv_d2  = 1.0f/CameraCenter2.at<float>(2);//depth

  //Find epipoles // The image in one view of the camera centre of the other view
  float ex_C1in2 = pKF2->mfx*CameraCenter2.at<float>(0)*inv_d2+pKF2->mcx;
  float ey_C1in2 = pKF2->mfy*CameraCenter2.at<float>(1)*inv_d2+pKF2->mcy;

  //Find Matches via ORB Vocabulary
  int nNumMatches = 0;
  vector<bool> vbMatchedInKF2(pKF2->mnNumKP,false);
  vector<int> vMatches_1and2(pKF1->mnNumKP,-1);

  vector<int> counter[NumOfBins];
  for(int i=0; i<NumOfBins; i++)
    counter[i].reserve(500);

  DBoW2::FeatureVector::iterator itKF1 = vFeature_Vec1.begin();
  DBoW2::FeatureVector::iterator endKF1 = vFeature_Vec1.end();
  DBoW2::FeatureVector::iterator itKF2 = vFeature_Vec2.begin();
  DBoW2::FeatureVector::iterator endKF2 = vFeature_Vec2.end();


  while(itKF1 != endKF1 && itKF2 != endKF2)
  {
    //it is possible that the features match only when they are in the same node.
    if(itKF1->first == itKF2 -> first)//campare them feature points within the same node
    {
      //Get ID of KP
      vector<unsigned int> vIndexKF2 = itKF2 -> second;
      vector<unsigned int> vIndexKF1 = itKF1 -> second;
      //go through all the KPs in KF

      for(size_t i_KF1 =0; i_KF1<vIndexKF1.size(); i_KF1++)
      {
        //get the ID for KP
        size_t ID_KPKF1 = vIndexKF1[i_KF1];

        //Use the ID to access the MP
        MapPoint* pMP1 = pKF1->GetMP(ID_KPKF1);


        if(pMP1)//already a MP//skip
          continue;

        bool rightAvail1 = pKF1->mvDepthWithKP[ID_KPKF1]>=0;


        //Descriptor in KF
        const cv::Mat &d_KF1 = pKF1->mDescriptor.row(ID_KPKF1);
        //Get the KP
        cv::KeyPoint &kp1 = pKF1->mvUndisKP[ID_KPKF1];


        //Threshold
        int nBestDistance = LOWER_THRESHOLD;
        //Initialize ID
        int nBestKpID_KF2 = -1;

        //campare with KP in KeyFrame 2

        for(size_t i_KF2 = 0; i_KF2<vIndexKF2.size(); i_KF2++)
        {
          //get KP_ID in KeyFrame2
          size_t ID_KPKF2 = vIndexKF2[i_KF2];
          MapPoint* pMP2 = pKF2->GetMP(ID_KPKF2);

          //If the keypoint has been matched or the MP has existed
          if(vbMatchedInKF2[ID_KPKF2] || pMP2)
            continue;

          bool rightAvail2 = pKF2->mvDepthWithKP[ID_KPKF2]>=0;

          //Get descriptor
          const cv::Mat &d_KF2 = pKF2->mDescriptor.row(ID_KPKF2);

          //Compute difference
          int ndiff = DescriptorDifference(d_KF1, d_KF2);

          if (ndiff>LOWER_THRESHOLD || ndiff > nBestDistance)
            continue;


          //Find KP2
          cv::KeyPoint &kp2 = pKF2->mvUndisKP[ID_KPKF2];

          if(!rightAvail1 && !rightAvail2)
          {
            float dist_x = ex_C1in2-kp2.pt.x;
            float dist_y = ey_C1in2 -kp2.pt.y;

            float dist_sqr = dist_x * dist_x + dist_y * dist_y;
            //If distance is small, the kp2 is too close to C1
            if(dist_sqr<100*pKF2->mvScaleFactor[kp2.octave])
              continue;
          }

          //Evaluate through Epipolar Geometry
          if(EpipolarCheck(kp1,kp2,F_2_1,pKF2))
          {
            nBestKpID_KF2 = ID_KPKF2;
            nBestDistance = ndiff;
          }

        }

        if(nBestKpID_KF2>=0)
        {
          //vbMatchedInKF2[nBestKpID_KF2]=true;
          vMatches_1and2[ID_KPKF1] = nBestKpID_KF2;
          nNumMatches++;

        }
      }

      itKF1++;
      itKF2++;
    }
    else if(itKF1->first < itKF2->first)
    {
      //Find the pointer to smaller node in Frame
      itKF1 = vFeature_Vec1.lower_bound(itKF2->first);
    }
    else
    {
      itKF2= vFeature_Vec2.lower_bound(itKF1->first);
    }
  }

  //Output Matches
  vMatches.clear();
  vMatches.reserve(nNumMatches);

  for(size_t i = 0; i<vMatches_1and2.size();i++)
  {
    if(vMatches_1and2[i]<0) // i -> index of KP in KFq, vMatches_1and2[i] -> index of matched KP in KF2
      continue;
    vMatches.push_back(make_pair(i,vMatches_1and2[i]));
  }
  return nNumMatches;
}

//KF with Frame
int KeypointMatcher::MatchUsingBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vMatchedMP)
{

  int nNumMatches = 0;

  //Estimate the matching to remove error
  //collect the orientation of the pair of KeyPoints
  vector<int> counter[NumOfBins];
  //The counter counts the angle differnce between the Last KP and the best mathced current KP
  //initialize memory for each single bin
  for(int i=0; i<NumOfBins; i++)
  {
    counter[i].reserve(500);
  }

  //Get MPs in KF
  vector<MapPoint*> vMapPointInKF = pKF->GetAllMapPoints();
  //Initialize vector
  vMatchedMP = vector<MapPoint*>(F.mNumKeypoints,static_cast<MapPoint*>(NULL));

  DBoW2::FeatureVector &vFVectorKF = pKF->mFVector;
  DBoW2::FeatureVector &vFVectorF = F.mFVector;

  DBoW2::FeatureVector::iterator itKF = vFVectorKF.begin();
  DBoW2::FeatureVector::iterator endKF = vFVectorKF.end();
  DBoW2::FeatureVector::iterator itF = vFVectorF.begin();
  DBoW2::FeatureVector::iterator endF = vFVectorF.end();

  //For each feature vector, the first element is id node, the second element is ID for feature point
  //campare each feature
  while(itKF != endKF && itF != endF)
  {
    //it is possible that the features match only when they are in the same node.
    if(itKF->first == itF -> first)//campare them feature points within the same node
    {
      //Get ID of KP
      vector<unsigned int> vIndexF = itF -> second;
      vector<unsigned int> vIndexKF = itKF -> second;
      //go through all the KPs in KF

      for(size_t i_KF =0; i_KF<vIndexKF.size(); i_KF++)
      {
        //get the ID for KP
        unsigned int ID_KPKF = vIndexKF[i_KF];

        //Use the ID to access the MP
        MapPoint* pMP = vMapPointInKF[ID_KPKF];


        if(!pMP)
          continue;

        if(pMP->BadMP())
          continue;


        //Descriptor in KF
        const cv::Mat &d_KF = pKF->mDescriptor.row(ID_KPKF);


        //Totally different
        int nBestDistance = 256;
        int nBestDistance2 =256;
        //Initialize ID
        int nBestMatchedKP_ID = -1;

        //campare with KP in Frames

        for(size_t i_F = 0; i_F<vIndexF.size(); i_F++)
        {
          //get KP_ID in Frame
          unsigned int ID_KP = vIndexF[i_F];

          //Get descriptor
          const cv::Mat &d_F = F.mDescriptors.row(ID_KP);

          //Compute difference
          int ndiff = DescriptorDifference(d_KF, d_F);

          //Replace if satisfied
          if(ndiff<nBestDistance)
          {
            nBestDistance2 = nBestDistance;
            nBestDistance = ndiff;
            nBestMatchedKP_ID = ID_KP;
          }
          else if(ndiff<nBestDistance2)
          {
            nBestDistance2 = ndiff;
          }
        }

        //If it is below the setting threshold
        if(nBestDistance<=LOWER_THRESHOLD)
        {
          if((float)nBestDistance<mfRatio*(float)nBestDistance2)
          {
            //Update Matched MP
            vMatchedMP[nBestMatchedKP_ID] = pMP;

            //For each matched point, collect the orientation of the pair of KeyPoints
            float diff_angle = pKF->mvUndisKP[ID_KPKF].angle - F.mvUndisKP[nBestMatchedKP_ID].angle;
            if(diff_angle<0)
            {
              //adjust the angle diff
              diff_angle = diff_angle+360.0f;
            }

            //calculate which group the point should be
            int IndexForBin = round(diff_angle * (1.0f / NumOfBins));
            if(IndexForBin==NumOfBins)
              IndexForBin = 0;
            //put it into the counter
            if(IndexForBin < NumOfBins && IndexForBin >=0)
            {
              //i is for the index of the counter
              //in each index, it contains the ID of matched KP
              counter[IndexForBin].push_back(nBestMatchedKP_ID);
            }
            nNumMatches++;
          }

        }
      }
      itF++;
      itKF++;
    }else if(itF->first < itKF->first)
    {
      //Find the pointer to smaller node in Frame
      itF = vFVectorF.lower_bound(itKF->first);
    }
    else
    {
      itKF = vFVectorKF.lower_bound(itF->first);
    }
  }

  int i1 = -1; int i2 = -1; int i3 = -1;
  //find three orientation with the maximum counted value (Major orientation)
  ComputeMajorOrientation(counter,i1,i2,i3);
  //if the difference angle of the matched pair is not within the three types of orientation, delete
  for(int i = 0; i<NumOfBins; i++)
  {
    if(i!=i1 && i!=i2 && i!=i3)
    {
      for(int j = 0; j<counter[i].size(); j++)
      {
        //acess the assigned KP to delete
        //counter[i][j] to acess the ID of matched KP, i.e. the KP is not consistent with the major orientation
        vMatchedMP[counter[i][j]] = static_cast<MapPoint*>(NULL);
        nNumMatches--;
      }
    }
  }


  return nNumMatches;
}

//Find redundant MPs
int KeypointMatcher::FindRedundantMPs(KeyFrame *pKF, vector<MapPoint*> &vpMPs, float r)
{
  //Used for reprojection
  cv::Mat R_w_KF = pKF->GetRotationMatrix();
  cv::Mat t_w_KF = pKF->GetTranslationMatrix();
  //Get Camera parameters
  const float &fx = pKF->mfx;
  const float &fy = pKF->mfy;
  const float &cx = pKF->mcx;
  const float &cy = pKF->mcy;
  const float &b = pKF->mb;

  cv::Mat CC_KF = pKF->CameraCentre();

  int numFusedMPs = 0;

  //Go through all the MPs for reprojection to the KF
  for(int i=0; i<vpMPs.size(); i++)
  {
    MapPoint* pMP = vpMPs[i];

    if(!pMP)
      continue;

    if(pMP->BadMP() || pMP->inKeyFrame(pKF))
      continue;

    //Get world position of the MP
    cv::Mat MP_3Dw = pMP->GetWorldPosition();
    //Project the MP to the KF
    cv::Mat MP_3DKF = R_w_KF*MP_3Dw + t_w_KF;

    //Check depth
    if(MP_3DKF.at<float>(2)<0.0)
      continue;

    //Project it to pixel coordinate
    float x = MP_3DKF.at<float>(0);
    float y = MP_3DKF.at<float>(1);


    float inv_z = 1/MP_3DKF.at<float>(2);
    float u = fx*x*inv_z + cx;
    float v = fy*y*inv_z + cy;
    float u_r = u-b*inv_z;

    //Point inside the Image?
    if(!pKF->InImage(u,v))
      continue;

    //Check the scale
    float minDist=pMP->GetMinDist();
    float maxDist=pMP->GetMaxDist();

    //Dir
    cv::Mat MP2CC = MP_3Dw-CC_KF;
    //Distance
    float dist_MP2CC = cv::norm(MP2CC);
    if(dist_MP2CC<minDist || dist_MP2CC>maxDist)
      continue;
    cv::Mat MP_AvgDir = pMP->GetViewDir();

    //View angle should be less than 60
    float cos = MP2CC.dot(MP_AvgDir)/dist_MP2CC;
    if(cos<0.5)
      continue;

    int nPredictLevel  = pMP->PredictLevel(dist_MP2CC,pKF);
    //Search MPs in correspongding areas
    r = r*pKF->mvScaleFactor[nPredictLevel];

    //Indices of satisfied KPs
    vector<size_t> vIndices = pKF->FeaturesInArea(u,v,r);

    if(vIndices.empty())
      continue;

    //find best matched KP

    //Totally different
    int nBestDistance = 256;
    //Initialize ID
    size_t nBestMachedKP_ID = -1;

    const cv::Mat d_MP = pMP->GetDescriptor();
    //Go through each MPs which is in the relative area
    for(vector<size_t>::iterator it=vIndices.begin(), end=vIndices.end(); it!=end; it++)
    {
      size_t id_matched = *it;
      cv::KeyPoint &kp = pKF->mvUndisKP[id_matched];

      //Check scale
      int &KPLevel = kp.octave;
      if(KPLevel<nPredictLevel-1 || KPLevel>nPredictLevel)
        continue;

      //Check reprojection error
      int scale_square = pKF->mvScaleFactor[KPLevel]*pKF->mvScaleFactor[KPLevel];
      //if the right coordinates is in the view of the KF
      if(pKF->mvDepthWithKP[id_matched]>=0)
      {

        float &KP_x = kp.pt.x;
        float &KP_y = kp.pt.y;
        float &KP_r = pKF->mvDepthWithKP[id_matched];

        const float &error_x = u-KP_x;
        const float &error_y = v-KP_y;
        const float &error_rx = u_r-KP_r;

        float error = error_x*error_x+error_y*error_y+error_rx*error_rx;

        if(error>7.8*scale_square)
          continue;
      }else
      {
        float &KP_x = kp.pt.x;
        float &KP_y = kp.pt.y;

        const float &error_x = u-KP_x;
        const float &error_y = v-KP_y;
        const float error = error_x*error_x+error_y*error_y;

        if(error>5.991*scale_square)
          continue;
      }
      const cv::Mat &d_KF = pKF->mDescriptor.row(id_matched);
      int dist = DescriptorDifference(d_MP,d_KF);

      if(dist<nBestDistance)
      {
        nBestDistance = dist;
        nBestMachedKP_ID = id_matched;
      }


    }
    if(nBestDistance<=LOWER_THRESHOLD)
    {

      //find the matched MP in the KF
      MapPoint* pMP_KF = pKF->GetMP(nBestMachedKP_ID);
      if(pMP_KF)
      {
        if(!pMP_KF->BadMP())
        {
          //Choose the MP with more Obs
          if(pMP_KF->GetNumOfObs()>pMP->GetNumOfObs())
            pMP->ReplaceWith(pMP_KF);
          else
            pMP_KF->ReplaceWith(pMP);
        }
      }else{//Not MP
        pMP->AddMPObservation(pKF,nBestMachedKP_ID);
        pKF->AddMapPointToKF(pMP, nBestMachedKP_ID);
      }
      numFusedMPs++;
    }
  }
  return numFusedMPs;

}


int KeypointMatcher::DescriptorDifference(const cv::Mat &LastDescriptor, const cv::Mat &CurrentDescriptor)
{
  int dist = 0;
  //The descriptor is 32-byte length
  //int_32t:integer type with width of exactly 32 bits
  //Get the first item in the Mat that is the first 8 bits in 32-byte descriptor
  //initialize it as 32 bit for the operation later
  const int *pLast = LastDescriptor.ptr<int32_t>();//each holds 32 bit
  const int *pCurrent = CurrentDescriptor.ptr<int32_t>();//each holds 32 bit

  for(int i=0; i<8;i++, pLast++, pCurrent++)
  {
    int DifferentBit = *pLast ^ *pCurrent;//get different bits
    //count in 2-bit chunk
    DifferentBit = DifferentBit - ((DifferentBit>>1)&0x55555555);
    //count in 4-bit chunk of the result of 2-bit chunk
    DifferentBit = (DifferentBit&0x33333333)+((DifferentBit>>2)&0x33333333);
    //count in 8-bit chunk
    DifferentBit = (DifferentBit&0x0F0F0F0F)+((DifferentBit>>4)&0x0F0F0F0F);

    //ADD 8-bit chunks
    DifferentBit = DifferentBit + (DifferentBit >> 8);
    DifferentBit = DifferentBit + (DifferentBit >>16);
    //apply mask
    dist += (DifferentBit & 0x0000003F);

  }
  return dist;
}

void KeypointMatcher::ComputeMajorOrientation(vector<int> *counter, int &max_i1, int &max_i2, int &max_i3)
{
  //value to record the amount
  int max_1 =0; int max_2=0; int max_3 = 0;
  //access each bin to count the num of the element
  for(int i = 0; i<NumOfBins; i++)
  {
    if(counter[i].size()>max_1)
    {
      //update value for recording the amount
      max_3 = max_2;
      max_2 = max_1;
      max_1 = counter[i].size();
      //update value for index
      max_i3 = max_i2;
      max_i2 = max_i1;
      max_i1 = i;
    }
    else if(counter[i].size()>max_2)
    {
      //update value for recording the amount
      max_3 = max_2;
      max_2 = counter[i].size();
      //update the value for index
      max_i3 = max_i2;
      max_i2 = i;
    }
    else if(counter[i].size()>max_3)
    {
      //update the value for recording the amount
      max_3 = counter[i].size();
      //update the value for index
      max_i3 = i;
    }
  }
  //after sorting the records, delete some of them which does not meet the satisfaction
  if((float)max_1*0.1f > max_2)
  {
    //delete the record
    max_i2 = -1;
    max_i3 = -1;
  }
  else if((float)max_1*0.1f > max_3)
  {
    max_i3 = -1;
  }
}


bool KeypointMatcher::EpipolarCheck(cv::KeyPoint &kp1, cv::KeyPoint &kp2, cv::Mat &F_2_1, KeyFrame *pKF2)
{
  //Constructing the epipolar line l = x1'F (l=[a b c]T)
  //Epipolar line in the second view (camera 1)
  float a = kp1.pt.x*F_2_1.at<float>(0,0)+kp1.pt.y*F_2_1.at<float>(1,0)+F_2_1.at<float>(2,0);
  float b = kp1.pt.x*F_2_1.at<float>(0,1)+kp1.pt.y*F_2_1.at<float>(1,1)+F_2_1.at<float>(2,1);
  float c = kp1.pt.x*F_2_1.at<float>(0,2)+kp1.pt.y*F_2_1.at<float>(1,2)+F_2_1.at<float>(2,2);


  //l : ax + by + c = 0;
  //the distance of point(m,n) to l is |am + bn + c| / sqrt(a^2 + b^2)
  float top = a*kp2.pt.x + b*kp2.pt.y+c;
  float bottom = a*a + b*b;

  if(bottom == 0)
    return false;

  float square_result = top*top / bottom;

  float squre_scale = pKF2->mvScaleFactor[kp2.octave] * pKF2->mvScaleFactor[kp2.octave];//small -> small error

  bool ok = square_result < 3.84 * squre_scale;

  return ok;
}


//Define the searching radius by the cos calculated in frame class
float KeypointMatcher::DefineRadiusByCos(float &viewCos)
{
  //cos ~ 1 --> angle ~ 0 ---> search in a small area
  if(viewCos>=0.98)
    return 2.5;
  else
    return 4;
}



}//end of namespace
