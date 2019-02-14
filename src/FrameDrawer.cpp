#include "FrameDrawer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

namespace ORB_SLAM
{
//constructor
FrameDrawer::FrameDrawer(Map *pMap):mpMap(pMap)
{
  mState = Tracking::NOT_READY;
  //initialize an Mat for drawing
  //size of the mat is 480*640
  mImage = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame()
{    
  cv::Mat image;
  int CurrentState;
  //Keypoints of current frame
  vector<cv::KeyPoint> vCurrentKP;
  //New added MP (constructed by frame only)
  vector<bool> vMap_NewMP;
  //matched MP
  vector<bool> vMap_MP;

{
  unique_lock<mutex> Lock(mMutex);

  CurrentState = mState;
  if(mState == Tracking::NOT_READY)
  {
    mState = Tracking::NO_IMAGE;
  }
  mImage.copyTo(image);

  if(mState==Tracking::NOT_INITIALIZE)
  {
    vCurrentKP = vCurrentKP = mvCurrentKP;

  }
  else if(mState == Tracking::READY)
  {
    vCurrentKP = mvCurrentKP;
    vMap_MP = mvMap_MP;
    vMap_NewMP = mvMap_NewMP;
  }
  else if(mState==Tracking::LOST)
  {
    vCurrentKP = mvCurrentKP;
  }
} //Release lock
  if(image.channels()<3)
  {
    //transform graysacle image to BRG image
    cvtColor(image,image,CV_GRAY2BGR);
  }

  //Draw
  //initialize
  if(CurrentState == Tracking::READY)
  {
    //Draw keypoints
    mnTracking = 0;
    mnTrackedNew=0;
    int n = vCurrentKP.size();
    float r = 5.0;
    for(int i=0; i<n; i++)
    {
      if(vMap_MP[i] ||vMap_NewMP[i])
      {
        //pick two nearby points of the features eg.
       cv::Point2f point1,point2;
       point1.x = vCurrentKP[i].pt.x-r;
       point1.y = vCurrentKP[i].pt.y-r;
       point2.x = vCurrentKP[i].pt.x+r;
       point2.y = vCurrentKP[i].pt.y+r;

       if(vMap_MP[i])//For MP has Observations
       {
         //void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
         cv::rectangle(image,point1,point2,cv::Scalar(0,255,0));
         mnTracking++;
       }
       else//Draw new added MPs, these MPs can only be observed by current frame
           //These MPs are costructed by frame, not KeyFrame
       {
         cv::rectangle(image,point1,point2,cv::Scalar(255,0,0));
         mnTrackedNew++;

       }
      }
    }
  }

  cv::Mat imInfo;
  DrawInfo(image, CurrentState, imInfo);
 //return the Infomation
  return imInfo;
}

void FrameDrawer::DrawInfo(cv::Mat &image, int state, cv::Mat &imInfo)
{
  stringstream sstr;
  if( state==Tracking::NO_IMAGE)
  {
    sstr << "Waiting for image";
  }
  else if(state == Tracking::NOT_INITIALIZE)
  {
    sstr << "Waiting for Initialize";
  }
  else if( state == Tracking::READY)
  {

    sstr << "KeyFrames: " << mpMap->GetNumOfKF() << ", MapPoints: "<< mpMap->GetNumOfMP() << " Matches: " << mnTracking;

  }
  else if( state == Tracking::LOST)
  {
    sstr << "Tracking Lost! ";
  }


  int baseline = 0;

  //Size getTextSize(const string& text, int fontFace, double fontScale, int thickness, int* baseLine)
  //----------------------------
  //text – Input text string.
  //text_string – Input text string in C format.
  //fontFace – Font to use. See the putText() for details.
  //fontScale – Font scale. See the putText() for details.
  //thickness – Thickness of lines used to render the text. See putText() for details.
  //baseLine – Output parameter - y-coordinate of the baseline relative to the bottom-most text point.
  cv::Size size = cv::getTextSize(sstr.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

  imInfo = cv::Mat(image.rows+size.height+10,image.cols,image.type());
  image.copyTo(imInfo.rowRange(0,image.rows).colRange(0,image.cols));
  imInfo.rowRange(image.rows,imInfo.rows) = cv::Mat::zeros(size.height+10,image.cols,image.type());

  //void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
  //----------------
  //img – Image.
  //text – Text string to be drawn.
  //org – Bottom-left corner of the text string in the image.
  //font – CvFont structure initialized using InitFont().
  //fontFace – Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID’s can be combined with FONT_ITALIC to get the slanted letters.
  //fontScale – Font scale factor that is multiplied by the font-specific base size.
  //color – Text color.
  //thickness – Thickness of the lines used to draw a text.
  //lineType – Line type. See the line for details.
  //bottomLeftOrigin – When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
  cv::putText(imInfo,sstr.str(),cv::Point(5,imInfo.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
}


//Process the last tracked frame
void FrameDrawer::Update(Tracking *pTracking)
{
  unique_lock<mutex> Lock(mMutex);  //When updating the INFO, cannot draw it and vice versa

  //Copy the tracked image
  pTracking ->mGray.copyTo(mImage);
  //Copy the tracked the feature points in frame.
  mvCurrentKP = pTracking->mCurrentFrame.mvKeyPoints;

  mNumKeyPoint = mvCurrentKP.size();

  //Initialize vector
  //For MP IN MP
  mvMap_MP = vector<bool>(mNumKeyPoint, false);
  //For new added MP
  mvMap_NewMP = vector<bool>(mNumKeyPoint, false);

  //The tracker is ready
  if(pTracking -> mLastState == Tracking::READY)
  {
    for(int i=0; i<mNumKeyPoint; i++)
    {
      MapPoint* pMP = pTracking -> mCurrentFrame.mvMapPoint[i];
      if(pMP)
      {
        //within seen area
        if(!pTracking->mCurrentFrame.mvOutliers[i])
        {
          //If the MP can be seen by frames
          if(pMP->GetNumOfObs()>0)
          {
            mvMap_MP[i]=true;//This MP in Map is OK

          }
          else
          {
            mvMap_NewMP[i]=true;//MP constructed by frame
          }
        }
      }
    }
  }
  mState=static_cast<int>(pTracking->mLastState);
}

}//end of namespace
