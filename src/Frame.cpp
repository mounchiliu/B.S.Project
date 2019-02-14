

#include "Frame.h"


namespace ORB_SLAM{

//initialize
long unsigned int Frame::LastId = 0;
bool Frame::mbInitialComputations = true;//for first frame
float Frame::cx, Frame::cy, Frame::fx, Frame::fy;
float Frame::invfx, Frame::invfy;
float Frame::mnMax_X, Frame::mnMax_Y, Frame::mnMin_X, Frame::mnMin_Y;
float Frame::mfGridIndividualHeight, Frame::mfGridIndividualWidth;

Frame::Frame(const Frame &frame)
     : mpORBextractor(frame.mpORBextractor),mTimeStamp(frame.mTimeStamp), mCalibrationMatrix(frame.mCalibrationMatrix.clone()),
     mDistortCoef(frame.mDistortCoef.clone()),mbfx(frame.mbfx), mb(frame.mb), mThreshDepth(frame.mThreshDepth),
     mNumKeypoints(frame.mNumKeypoints), mvKeyPoints(frame.mvKeyPoints), mvUndisKP(frame.mvUndisKP), mvDepth(frame.mvDepth),
     mvDepthWithKPU(frame.mvDepthWithKPU), mDescriptors(frame.mDescriptors.clone()), mnId(frame.mnId),
     mnScaleLevels(frame.mnScaleLevels),mfScaleFactor(frame.mfScaleFactor), mvScaleFactors(frame.mvScaleFactors),
     mvInvScaleFactors(frame.mvInvScaleFactors), mpVocabulary(frame.mpVocabulary), mFVector(frame.mFVector),
     mBOWVector(frame.mBOWVector),mvMapPoint(frame.mvMapPoint), mvOutliers(frame.mvOutliers),
     mpReferenceFrame(frame.mpReferenceFrame)
{
  for(int i=0;i<GRID_COLS;i++)
  {
      for(int j=0; j<GRID_ROWS; j++)
      {
          mvGrid[i][j]=frame.mvGrid[i][j];
      }
  }

  if(!frame.mTWorld2Cam.empty())
      UpdatePose(frame.mTWorld2Cam);
}




Frame::Frame(const cv::Mat &image_Gray, const cv::Mat &image_Depth,
             const double &timeStamp, ORBextractor* pextractor, ORBVocabulary* pVocabulary, cv::Mat &CalibrationMatrix,
             cv::Mat &distortCoef, const float &bfx, const float &thDepth)
  :mTimeStamp(timeStamp),mpORBextractor(pextractor), mpVocabulary(pVocabulary), mCalibrationMatrix(CalibrationMatrix.clone()),mDistortCoef(distortCoef.clone()),
    mbfx(bfx), mThreshDepth(thDepth)
{
    //Frame ID. Once obtain a frame, id+1
    mnId = LastId++;



    // Scale Level Info
    mnScaleLevels = mpORBextractor->GetLevels();
    mfScaleFactor = mpORBextractor->GetScaleFactor();    
    mvScaleFactors = mpORBextractor->GetScaleFactors();
    mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();

    // ORB extraction
    ExtractORB(image_Gray);


    mNumKeypoints = mvKeyPoints.size();

    //No keypoints, return
    if(mvKeyPoints.empty())
        return;

    //Initialize MP
    mvMapPoint = vector<MapPoint*>(mNumKeypoints,static_cast<MapPoint*>(NULL));
    //Initialize outliers MP
    mvOutliers = vector<bool>(mNumKeypoints, false);


    //Remove the distortion for Keypoints
    UndistortKeyPoints();

    //Associate right coordinate to a key point
    ComputeStereoFromRGBD(image_Depth);


    //For first Frame
    if(mbInitialComputations)
     {
      ComputeImageBounds(image_Gray);
      //GRID_ROWS 54 //number of rows
      //GRID_COLS 96 //number of cols
      mfGridIndividualWidth = static_cast<float> (GRID_COLS) / static_cast<float> (mnMax_X - mnMin_X);
      mfGridIndividualHeight = static_cast<float> (GRID_ROWS) / static_cast<float> (mnMax_Y - mnMin_Y);

      //Calibration Matrix and parameters
      //Set the parameter via Calibration Matrix
      fx = CalibrationMatrix.at<float>(0,0);
      fy = CalibrationMatrix.at<float>(1,1);
      cx = CalibrationMatrix.at<float>(0,2);
      cy = CalibrationMatrix.at<float>(1,2);
      invfx = 1.0f / fx ;
      invfy = 1.0 / fy;

      //Finish initialization
      mbInitialComputations = false;
    }

    mb = mbfx / fx;

    AssignFeaturesToGrid();

}


void Frame::ExtractORB(const cv::Mat &image)
{
  //From opencv ORBextractor.cpp
  //void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
    (*mpORBextractor)(image,cv::Mat(),mvKeyPoints,mDescriptors);
}



void Frame::FindBOW()
{

  if(mBOWVector.empty() || mFVector.empty())
  {
    //convert the MAT of descriptor to vector
    vector<cv::Mat> vDesc = DescriptorVector(mDescriptors);
    //mBOWvector -> (Output) bow vector
    //mFvector -> (Output) feature vector of nodes and feature indexes
    //levelsup ->	levels to go up the vocabulary tree to get the node index
    mpVocabulary->transform(vDesc,mBOWVector,mFVector,4);

  }

}


cv::Mat Frame::GetCameraCentre()
{
  return mtCam2world.clone();
}


cv::Mat Frame::GetInverseRotation()
{
  return mRCam2World.clone();
}

vector<cv::Mat> Frame::DescriptorVector(const cv::Mat &Descriptor)
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

void Frame::AssignFeaturesToGrid()
{

  //#define GRID_ROWS 48
  //#define GRID_COLS 64
  int nReserve = (0.5f*mNumKeypoints)/(GRID_COLS * GRID_ROWS);
  //initialize memory
  for(int i = 0; i < GRID_COLS; i++)
  {
    for(int j = 0; j < GRID_ROWS; j++)
    {
      mvGrid[i][j].reserve(nReserve);
    }
  }

  for(int i = 0; i<mNumKeypoints; i++)
  {
    const cv::KeyPoint &kp = mvUndisKP[i];

    int nGridPosX, nGridPosY;
    nGridPosX = round((kp.pt.x - mnMin_X)*mfGridIndividualWidth);
    nGridPosY = round((kp.pt.y - mnMin_Y)*mfGridIndividualHeight);

    //if the keypoint is in the grid
    if(InGrid(nGridPosX,nGridPosY))
    {
      mvGrid[nGridPosX][nGridPosY].push_back(i);
      //at the position of the KP, mvGrid hold the index of the KP
    }
  }
}

vector<size_t> Frame::FeaturesInArea(const float &x, const float &y, const float &r, const int minLevelOfPyramid, const int maxLevelOfPyramid) const
{
  //Initialize a vector to store the index of satisfied KeyPoint
  vector<size_t> vIndex;
  vIndex.reserve(mNumKeypoints);

  //mfGridIndividualWidth represents the number of cells for each pixel in a row
  //mfGridIndividualHeight represents the number of cells for each pixel in a col
  //round downward the value
  int nMinCellInX = max(0,(int)floor((x-mnMin_X-r)*mfGridIndividualWidth));  // should be greater than 0
  //shoud not exceed the num of cols defined before
  if(nMinCellInX>=GRID_COLS)
    return vIndex; // the point does not satisfy the requirement

  //round up the value
  //use (GRID_COLS-1) because we need to define a square for searching
  int  nMaxCellInX = min((int)GRID_COLS-1,(int)ceil((x-mnMin_X+r)*mfGridIndividualWidth));
  //should not smaller than 0
  if(nMaxCellInX<0)
    return vIndex;// the point does not satisfy the requirement

  //Similar to the y axis
  //round downward the value
  int nMinCellInY = max(0,(int)floor((y-mnMin_Y-r)*mfGridIndividualHeight));  // should be greater than 0
  //shoud not exceed the num of cols defined before
  if(nMinCellInY>=GRID_COLS)
    return vIndex; // the point does not satisfy the requirement

  //round up the value
  //use (GRID_COLS-1) because we need to define a square for searching
  int  nMaxCellInY = min((int)GRID_ROWS-1,(int)ceil((y-mnMin_Y+r)*mfGridIndividualHeight));
  //should not smaller than 0
  if(nMaxCellInY<0)
    return vIndex;// the point does not satisfy the requirement

  //check the defined level
  bool bCheck = (minLevelOfPyramid>0) || (maxLevelOfPyramid >=0);

  for(int temp_x = nMinCellInX; temp_x <= nMaxCellInX; temp_x++)
  {
    for(int temp_y = nMinCellInY; temp_y <= nMaxCellInY; temp_y++)
    {
      //Initialize a vector
      //pass the index of all KPs to this vector
      //this vector stores all the indices of KPs
      const vector<size_t> vIndexForKP = mvGrid[temp_x][temp_y];
      if(vIndexForKP.empty()) // If there is no KP at this position
        continue; //skip this condition

      for (size_t i = 0; i < vIndexForKP.size(); i++)
      {
        //Get the undistorted KP of Previous frame
        const cv::KeyPoint &KPU = mvUndisKP[vIndexForKP[i]];
        if(bCheck)
        {
          //octave (pyramid layer) from which the keypoint has been extracted
          if(KPU.octave<minLevelOfPyramid)
            continue; //skip this condition
          if(maxLevelOfPyramid>0)
            if(KPU.octave>maxLevelOfPyramid)
              continue;//skip this condition
        }

        //x,y -> coordinate of the KeyPoint of last frame projected in current frame
        //this coordinate will be campared with the keypoints in current frame using method below
        float dist_x = fabs(KPU.pt.x - x);
        float dist_y = fabs(KPU.pt.y - y);

        if(dist_x<r && dist_y<r)
        {
          //pass the index of satisfied keypoint
          //this vector only stores the indices of satisfied KP
          vIndex.push_back(vIndexForKP[i]);
        }
      }
    }
  }
   return vIndex;

}


void Frame::UpdatePose(cv::Mat TWorld2Cam)
{
    mTWorld2Cam = TWorld2Cam.clone();
    PoseMatrices();
}


void Frame::PoseMatrices()
{
    //mTWorld2Cam --> Camera pose (Transfer Matrix)
    mRWorld2Cam = mTWorld2Cam.rowRange(0,3).colRange(0,3);
    mRCam2World = mRWorld2Cam.t();
    mtWorld2Cam = mTWorld2Cam.rowRange(0,3).col(3);
    //camera centre
    mtCam2world = -mRCam2World*mtWorld2Cam;
}

bool Frame::InGrid(int &Pos_X, int &Pos_Y)
{
  //X , Y FOR NUM OF CELL FOR COL and ROW
  //Keypoint is in the Grid?
  return(Pos_X >= 0 && Pos_X < GRID_COLS && Pos_Y >= 0 && Pos_Y < GRID_ROWS);
}


void Frame::UndistortKeyPoints()
{
  //If the point has been corrected
  if(mDistortCoef.at<float>(0) == 0.0)
  {
    mvUndisKP = mvKeyPoints;
    return;
  }

  //mNumKeypoints is the number of keypoints
  //Build a mNumKeypoints*2 Mat to store the coordinate values of keypoint
  cv::Mat mat(mNumKeypoints,2,CV_32F);
  for(int i=0; i<mNumKeypoints; i++)
  {
    //In order to use function in opencv, the InputArray 'src' is observed point coordinates
    //InputArray 'src' shoul be 1*N or N*1 2-channal Mat
    //Therefore, split the Mat.
    mat.at<float>(i,0) = mvKeyPoints[i].pt.x;
    mat.at<float>(i,1) = mvKeyPoints[i].pt.y;
  }

  //Undistort keypoints
  //InputArray 'src' shoul be (1*N) or N*1 2-channal Mat
  mat = mat.reshape(2);
  //Computes the ideal point coordinates from observed point coordinates.
  //void cv::undistortPoints(InputArray src, OutputArray dst, InputArray cameraMatrix,
                             //inputArray disortCoeffs, InputArray R, InputArray P)
  //R--Rectification transformation in the object space. If the matrix is empty, the indentity transformation is used
  //P--New camera matrix.  If the matrix is empty, the indentity transformation is used
  cv::undistortPoints(mat,mat,mCalibrationMatrix,mDistortCoef,cv::Mat(),mCalibrationMatrix);
  mat = mat.reshape(1);

  //Store undistorted keypoints
  mvUndisKP.resize(mNumKeypoints);
  for(int i=0; i<mNumKeypoints; i++)
  {
    cv::KeyPoint keypoint = mvKeyPoints[i];
    keypoint.pt.x = mat.at<float>(i,0);
    keypoint.pt.y = mat.at<float>(i,1);
    mvUndisKP[i] = keypoint;
  }
}

void Frame::ComputeImageBounds(const cv::Mat &image)
{
  //if distorted
  if(mDistortCoef.at<float>(0)!=0.0)
  {
    //correct the points of four corners (0,0) (col,0) (0,col) (0,row) (col,row)
    cv::Mat imageForBounds(4,2,CV_32F);
    //upper left corner
    imageForBounds.at<float>(0,0) = 0.0; //value for x
    imageForBounds.at<float>(0,1) = 0.0; //value for y
    //upper right corner
    imageForBounds.at<float>(1,0) = image.cols;
    imageForBounds.at<float>(1,1) = 0.0;
    //left bottom
    imageForBounds.at<float>(2,0) = 0.0;
    imageForBounds.at<float>(2,1) = image.rows;
    //right bottom
    imageForBounds.at<float>(3,0) = image.cols;
    imageForBounds.at<float>(3,1) = image.rows;

    //Undistorted corners
    imageForBounds = imageForBounds.reshape(2);
    cv::undistortPoints(imageForBounds,imageForBounds,mCalibrationMatrix,mDistortCoef,cv::Mat(),mCalibrationMatrix);
    imageForBounds = imageForBounds.reshape(1);

    //Choose the smaller value between the x values of upper-left corner and left bottom
    mnMin_X = min(imageForBounds.at<float>(0,0),imageForBounds.at<float>(2,0));
    //Choose the larger value between the x values of upper-right corner and right bottom
    mnMax_X = max(imageForBounds.at<float>(1,0),imageForBounds.at<float>(3,0));
    //Choose the smaller value between the y values of upper-left corner and upper-right corner
    mnMin_Y = min(imageForBounds.at<float>(0,1),imageForBounds.at<float>(1,1));
    //Choose the larger value between the y values of left bottom and right bottom
    mnMax_Y = max(imageForBounds.at<float>(2,1),imageForBounds.at<float>(3,1));

  }
  else // if undistorted
  {
    /*
     * (0,row)      (col,row)
     *
     *
     * (0,0)        (col,0)
     */
    mnMin_X = 0.0f;
    mnMax_X = image.cols;
    mnMin_Y = 0.0f;
    mnMax_Y = image.rows;
  }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &image_Depth)
{


  //initialize
  mvDepthWithKPU = vector<float> (mNumKeypoints,-1);
  mvDepth = vector<float>(mNumKeypoints,-1);

  for(int i=0; i<mNumKeypoints; i++)
  {
    const cv::KeyPoint &keypoint = mvKeyPoints[i];
    const cv::KeyPoint &UndisKey = mvUndisKP[i];

    const float &y = keypoint.pt.y;//row
    const float &x = keypoint.pt.x;//col

    const float d = image_Depth.at<float>(y,x);

    if(d>0)
    {
      //depth
      mvDepth[i] = d;
      //Stereo disparity
      float disparity = mbfx/d;
      //associate the depth with keypoints
      //Calculate the x value of same point in right camera projected into the first camera
      mvDepthWithKPU[i] = UndisKey.pt.x - disparity;
      //If the value < 0 -> the point is not in the view of the left camera

    }

  }
}

cv::Mat Frame::BackProject(const int &i)
{
  //i is the index for KP
  //Get depth
  const float z = mvDepth[i];

  if(z>0)
  {
    const float u = mvUndisKP[i].pt.x;
    const float v = mvUndisKP[i].pt.y;
    const float x = (u-cx)*z*invfx;
    const float y = (v-cy)*z*invfy;
    cv::Mat Three_D =(cv::Mat_<float>(3,1)<<x,y,z);
    return mRCam2World*Three_D+mtCam2world;
  }
  else
    return cv::Mat();
}

bool Frame::MPInViewArea(MapPoint *pMP)
{
  //To record whether the MP Need Track and projection?
  pMP->mbShouldTrack = false;

  //Get World Position
  cv::Mat WorldPosOfMP = pMP->GetWorldPosition();

  //Project to the camera coordinate of current frame
  cv::Mat PosInCam = mRWorld2Cam * WorldPosOfMP + mtWorld2Cam;


  //Get coordinates of the point
  const float &Pc_X = PosInCam.at<float>(0);
  const float &Pc_Y = PosInCam.at<float>(1);
  const float &Pc_Z = PosInCam.at<float>(2);

  //check the depth
  if(Pc_Z < 0)
    return false;


  const float inv_Z = 1.0f / Pc_Z ;

  //coordinates for pixel coordinate
  const float u = fx * Pc_X * inv_Z + cx;
  const float v = fy * Pc_Y * inv_Z + cy;

  //check bounds
  if( u < mnMin_X || u > mnMax_X )
    return false;

  if( v < mnMin_Y || v > mnMax_Y)
    return false;

  //check distance
  const float dist_max = pMP->GetMaxDist();
  const float dist_min = pMP->GetMinDist();

  const cv::Mat Dir = WorldPosOfMP - mtCam2world;
  const float dist = cv::norm(Dir);

  if(dist>dist_max || dist<dist_min)
     return false;

  //check dir of view
  const cv::Mat dir_avg = pMP->GetViewDir(); //|dir| = 1

  //cosine of the angle of the current dir and the avg. dir
  const float cos = Dir.dot(dir_avg)/dist;

  //angle bigger than 60degree
  if(cos<0.5f)
    return false;

  //Predict level for this mp
  pMP->mnPredictedLevel = pMP->PredictLevel(dist,this);
  //Mark the MP should be tracked and projected
  pMP->mbShouldTrack = true;
  //Record the cos
  pMP->mfViewCos = cos;

  //Record the coordinates after the projection
  pMP->mfProj_X = u;
  pMP->mfProj_Y = v;
  pMP->mfProj_XR = u - mbfx*inv_Z;


  return true;
}











}//end of namespace

