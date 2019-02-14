#include "MapDrawer.h"



namespace ORB_SLAM{


MapDrawer::MapDrawer(Map* pMap, const string &SettingFile): mpMap(pMap)
{
  cv::FileStorage fSetting(SettingFile, cv::FileStorage::READ);
  //Settings for viewer
  mSizeOfPoint = fSetting["Viewer.SizeOfPoint"];
  //The radio between width of the camera and the total width
  mSizeOfCamera = fSetting["Viewer.SizeOfCamera"];
}

void MapDrawer::SetCameraPose(cv::Mat &CurrentPose)
{
  unique_lock<mutex> Lock(mMutex_Camera);
  mCameraPose = CurrentPose.clone();
}

void MapDrawer::GetCUrrentCameraPose(pangolin::OpenGlMatrix &Output_CameraMatrix)
{
  if(!mCameraPose.empty())
  {
    //Rotation Matrix
    cv::Mat R_c2w(3,3,CV_32F);
    //Translation Matrix
    cv::Mat t_c2w(3,1,CV_32F);
    {
      unique_lock<mutex> Lock(mMutex_Camera);
      R_c2w = mCameraPose.rowRange(0,3).colRange(0,3).t();
      t_c2w = -R_c2w*mCameraPose.rowRange(0,3).col(3);
    }
    //First Col
    Output_CameraMatrix.m[0] = R_c2w.at<float>(0,0);
    Output_CameraMatrix.m[1] = R_c2w.at<float>(1,0);
    Output_CameraMatrix.m[2] = R_c2w.at<float>(2,0);
    Output_CameraMatrix.m[3] = 0.0;
    //Second Col
    Output_CameraMatrix.m[4] = R_c2w.at<float>(0,1);
    Output_CameraMatrix.m[5] = R_c2w.at<float>(1,1);
    Output_CameraMatrix.m[6] = R_c2w.at<float>(2,1);
    Output_CameraMatrix.m[7]  = 0.0;
    //Third Col
    Output_CameraMatrix.m[8] = R_c2w.at<float>(0,2);
    Output_CameraMatrix.m[9] = R_c2w.at<float>(1,2);
    Output_CameraMatrix.m[10] = R_c2w.at<float>(2,2);
    Output_CameraMatrix.m[11]  = 0.0;
    //Fourth Col
    Output_CameraMatrix.m[12] = t_c2w.at<float>(0);
    Output_CameraMatrix.m[13] = t_c2w.at<float>(1);
    Output_CameraMatrix.m[14] = t_c2w.at<float>(2);
    Output_CameraMatrix.m[15]  = 1.0;
  }
}


void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &CameraPose)
{
  float ratio_w = mSizeOfCamera;
  float height = ratio_w*0.75;
  float weight = ratio_w*0.6;

  //pushes the current matrix stack down by one and duplicates the current matrix
  //the matrix on top of the stack = the one below it
  glPushMatrix();

  glMultMatrixd(CameraPose.m);

  //Draw lines between tow vertexes
  //Set width of the line
  glLineWidth(2);
  glColor3f(0.0,0.0,1.0);
  glBegin(GL_LINES);

  glVertex3f(0,0,0);
  glVertex3f(ratio_w,height,weight);

  glVertex3f(0,0,0);
  glVertex3f(ratio_w,-height,weight);

  glVertex3f(0,0,0);
  glVertex3f(-ratio_w,-height,weight);

  glVertex3f(0,0,0);
  glVertex3f(-ratio_w,height,weight);

  glVertex3f(ratio_w,height,weight);
  glVertex3f(-ratio_w,height,weight);

  glVertex3f(-ratio_w,height,weight);
  glVertex3f(-ratio_w,-height,weight);

  glVertex3f(-ratio_w,-height,weight);
  glVertex3f(ratio_w,-height,weight);

  glVertex3f(ratio_w,-height,weight);
  glVertex3f(ratio_w,height,weight);

  glEnd();

  glPopMatrix();

}

void MapDrawer::DrawKeyFrame()
{
 //Get the size for drawn kF
  float size = 0.05;
  float height = 0.75*size;
  float weight = 0.6*size;

  vector<KeyFrame*> vKFs = mpMap->GetKFs();

  for(size_t i=0; i<vKFs.size(); i++)
  {
    KeyFrame* pKF = vKFs[i];//get one KF
    //get transformation matrix
    cv::Mat T_c2w = pKF->GetInversePose().t();

    glPushMatrix();

    glMultMatrixf(T_c2w.ptr<GLfloat>(0));

    //Draw lines between tow vertexes
    //Similar to the task of drawing a camera
    //Set width of the line
    glLineWidth(2);
    glColor3f(1.0,0.0,1.0);
    glBegin(GL_LINES);

    glVertex3f(0,0,0);
    glVertex3f(size,height,weight);

    glVertex3f(0,0,0);
    glVertex3f(size,-height,weight);

    glVertex3f(0,0,0);
    glVertex3f(-size,-height,weight);

    glVertex3f(0,0,0);
    glVertex3f(-size,height,weight);

    glVertex3f(size,height,weight);
    glVertex3f(-size,height,weight);

    glVertex3f(-size,height,weight);
    glVertex3f(-size,-height,weight);

    glVertex3f(-size,-height,weight);
    glVertex3f(size,-height,weight);

    glVertex3f(size,-height,weight);
    glVertex3f(size,height,weight);

    glEnd();

    glPopMatrix();


  }


}

void MapDrawer::DrawMapPoint()
{
  //Get all the MPs
  const vector<MapPoint*> &vpMPsInMap = mpMap->GetMPs();
  //Get all Ref MPs in Map
  const vector<MapPoint*> &vpRefMPsInMap = mpMap->GetRefMPs();

  //In openGL, for drawing points
  //glPointSize(10.0f);  //set the point size to be 10 pixels in size
  //glColor3f(0.0f,0.0f,1.0f); //set blue color
  //glBegin(GL_POINTS); //starts drawing of points
  //  glVertex3f(1.0f,1.0f,0.0f);//Draw upper-right corner at point(1,1,0)
  //   glVertex3f(-1.0f,-1.0f,0.0f);//Draw lower-left corner at point(-1,-1,0)
  //glEnd();//end drawing of points

  //Draw MPs in Map except Ref MPs
  //SET POINT_SIZE = 2 pixels
  glPointSize(mSizeOfPoint);
  glBegin(GL_POINTS); //starts drawing points
  glColor3f(0.0, 0.0, 0.0);//Initialize color as black

  set<MapPoint*> sRefMPsInMap(vpRefMPsInMap.begin(),vpRefMPsInMap.end());


  //DRAW all the MPs except ref MPs in the Map as black points
  for(size_t i=0; i<vpMPsInMap.size(); i++)
  {
    if(vpMPsInMap[i]->BadMP() || sRefMPsInMap.count(vpMPsInMap[i]))//Skip Ref MPs
       continue;

    cv::Mat MPWorldPos = vpMPsInMap[i]->GetWorldPosition();
    //Draw this MP according to its position
    glVertex3f(MPWorldPos.at<float>(0), MPWorldPos.at<float>(1), MPWorldPos.at<float>(2));
  }
  glEnd();

  //Draw Ref MPs as Red (Ref MPs is the MPs in keyframes)
  glPointSize(mSizeOfPoint);
  glBegin(GL_POINTS);
  glColor3f(1.0,0.0,0.0);// RED

  for(size_t i=0; i<vpRefMPsInMap.size(); i++)
  {
    if(vpRefMPsInMap[i]->BadMP())
      continue;

    cv::Mat RefMPWorldPos = vpRefMPsInMap[i]-> GetWorldPosition();
    glVertex3f(RefMPWorldPos.at<float>(0), RefMPWorldPos.at<float>(1), RefMPWorldPos.at<float>(2));
  }
  glEnd();
}





}
