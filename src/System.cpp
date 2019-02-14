#include "System.h"
#include "Optimization.h"

#include <iostream>
using namespace std;

namespace ORB_SLAM
{


System::System(const string &SettingFile, const string &Vocabulary)
{
  cout << endl << "ORBSLAM" << endl;

  cout << endl << "Input sensor is KINECT(RGB-D camera)"<<endl;

  //load setting file
  cv::FileStorage fSettings(SettingFile.c_str(),cv::FileStorage::READ);
  if(!fSettings.isOpened())
  {
    cerr << "Failed to open the setting file" << endl;
    exit(-1);
  }

  float fResolution = fSettings["Resolution"];
  cout << "Resolution for PointCloud:"<<fResolution<<endl;

  //Load ORB Vocabulary
  cout << endl << "Loading ORB Vocabulary." << endl;

  mpORBVocabulary = new ORBVocabulary();
  bool bLoad = mpORBVocabulary->loadFromTextFile(Vocabulary);
  if(!bLoad)
  {
      cerr << "Falied to open: " << Vocabulary << endl;
      exit(-1);
  }
  cout << "Vocabulary loaded!" << endl << endl;



  mpMap = new Map();

  //Creat Drawers
  mpFrameDrawer = new FrameDrawer(mpMap);
  mpMapDrawer = new MapDrawer(mpMap,SettingFile);

  //Drawer for PointCloud
  mpPointCloudMap = make_shared<PointCloudMap>(fResolution);


  //Initialize Tracker
  mpTracking = new Tracking(mpFrameDrawer, mpMapDrawer, mpMap, mpPointCloudMap,SettingFile, mpORBVocabulary);

  //Initialize viewer and its thread
  mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracking, SettingFile);
  mptViewer = new thread(&Viewer::run, mpViewer);

  //Initialize the Local Map and its thread
  mpLocalMap = new LocalMap(mpMap);
  mptLocalMap = new thread(&LocalMap::run,mpLocalMap);

  //Set pointers
  mpTracking->SetViewer(mpViewer);
  mpTracking->SetLocalMap(mpLocalMap);

  mpLocalMap->SetTracking(mpTracking);
}



//Shut down the system
void System::ShutDownSystem()
{

  mpLocalMap->FinishRequset();
  mpPointCloudMap->ShutDown();

  if(mpViewer)
  {
      mpViewer -> SetFinishRequest();
      while(!mpViewer->Finished())
        usleep(5000);
  }

  while(!mpLocalMap->Finished())
    usleep(5000);

}

void System::SaveTrajectory(const string &file)
{
  cout << endl << "Camera Trajectory: " << file << endl;

  ofstream ffile;
  ffile.open(file.c_str());
  //fixed
  ffile << fixed;

  //Pose of Frames
  list<ORB_SLAM::KeyFrame*>::iterator lRefKF = mpTracking->mlRefKFs.begin();
  list<double>::iterator lT = mpTracking->mlTimeSample.begin();
  for(list<cv::Mat>::iterator lit=mpTracking->mlRecordedCurrRefInv.begin(),
      lend=mpTracking->mlRecordedCurrRefInv.end();lit!=lend;lit++, lRefKF++, lT++)
  {

      KeyFrame* pRefKF = *lRefKF;

      cv::Mat T_rw = cv::Mat::eye(4,4,CV_32F);

      // Keyframe was culled?
      // replace with a suitable keyframe.
      while(pRefKF->IsBad())
      {
          T_rw = T_rw*pRefKF->mTc_father;
          pRefKF = pRefKF->GetFather();
      }

      T_rw = T_rw*pRefKF->GetPose();

      cv::Mat T_cw = (*lit)*T_rw;
      cv::Mat R_wc = T_cw.rowRange(0,3).colRange(0,3).t();
      cv::Mat t_wc = -R_wc*T_cw.rowRange(0,3).col(3);

      vector<float> v_quat = Converter::toQuat(R_wc);

      ffile << setprecision(6) << *lT << " " <<  setprecision(9) << t_wc.at<float>(0)
            << " " << t_wc.at<float>(1) << " " << t_wc.at<float>(2)
            << " " << v_quat[0] << " " << v_quat[1] << " " << v_quat[2] << " " << v_quat[3] << endl;
  }

  //Close file
  ffile.close();

  cout << endl << "Trajectory Saved!" << endl;

}

void System::SaveKFTrajectory(const string &file)
{
  cout<< endl << "KeyFrame Trajactory: " <<file << endl;

  vector<KeyFrame*> vKFs = mpMap->GetKFs();
  //sort based on order (KF1.id < KF2.id)
  sort(vKFs.begin(),vKFs.end(),KeyFrame::Order);

  ofstream ffile;
  ffile.open(file.c_str());

  //fixed
  ffile << fixed;

  for(size_t i=0; i<vKFs.size(); i++)
  {
    KeyFrame* pKF = vKFs[i];

    // Keyframe was culled?
    // Skip
    if(pKF->IsBad())
      continue;

    cv::Mat Rotation = pKF->GetRotationMatrix().t();
    //Transfer to Quaternion
    vector<float> v_quat = Converter::toQuat(Rotation);
    cv::Mat CC = pKF->CameraCentre();

    ffile << setprecision(6) << pKF->mTimestamp << setprecision(7)
          << " " << CC.at<float>(0) << " " << CC.at<float>(1) << " " << CC.at<float>(2)
          << " " << v_quat[0] << " " << v_quat[1] << " " << v_quat[2] << " " << v_quat[3] <<endl;
  }

  //close file
  ffile.close();

  cout << endl << "KeyFrame Trajectory Saved! " << endl;

}



cv::Mat System::TrackRGB(const cv::Mat image, const cv::Mat &depth, const double &timestamp)
{
  return mpTracking->GrabRGB(image,depth,timestamp);
}

//Set pointers

}//end of namespace
