#include "Viewer.h"
#include <pangolin/pangolin.h>

namespace ORB_SLAM
{
Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer,MapDrawer* pMapDrawer, Tracking* pTracking, const string &Setting)
              :mpSys(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracking(pTracking),mbFinish(true),
               mbRequestFinish(false)

{
  //read Setting file to load Viewer Parameters
  cv::FileStorage fSetting(Setting, cv::FileStorage::READ);

  float fps = fSetting["Camera.fps"];
  mTime = 1e3/fps;

  mWidthOfImage = 640;
  mHeightOfImage = 480;
}

Viewer::~Viewer()
{
  cout<<"Deallocate Viewer"<<endl;
}

void Viewer::run()
{
  mbFinish = false;

  //Initialize pangolin  //Create Window
  pangolin::CreateWindowAndBind("Map",1024,768);

  //If enabled,do depth comparisons and update the depth buffer.
  //Hide points according to its distance to the camera
  //Use this to enable 3D Mouse handler
  glEnable(GL_DEPTH_TEST);
  //If enabled,blend two colors
  glEnable(GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  //Define Projection Matrix
  //ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
  //Initial ModelView Matrix
  //ModelViewLookAt (double ex, double ey, double ez, double lx, double ly, double lz, AxisDirection up)
  //Initial view point (ex ey ez)
  //looking at (lx,ly,lz)   (initial)
  //Initial view Direction: assumes forward is -z and up is +y
  pangolin::OpenGlRenderState s_cam(
              pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
              pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0)
              );

  //create interactive view
  pangolin::Handler3D handler(s_cam);
  //SetBounds (Attach bottom, Attach top, Attach left, Attach right, scale of width & height)
  pangolin::View &d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
          .SetHandler(new pangolin::Handler3D(handler));


  //Initialize Camera Pose
  pangolin::OpenGlMatrix T_c2w;
  T_c2w.SetIdentity();

  while(1)
  {
    //Draw Frame
    cv::Mat im = mpFrameDrawer->DrawFrame();
    cv::imshow("ORB-SLAM: Current Frame",im);
    cv::waitKey(mTime);


    //Draw map
    //clean buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mpMapDrawer->GetCUrrentCameraPose(T_c2w);

    //Activate
    d_cam.Activate(s_cam);

    //Set BG as white
    glClearColor(1.0,1.0,1.0,1.0);//white

    mpMapDrawer->DrawCurrentCamera(T_c2w);
    //mpMapDrawer->DrawTrajectory(T_c2w);
    mpMapDrawer->DrawMapPoint();
    mpMapDrawer->DrawKeyFrame();

    pangolin::FinishFrame();


    if(FinishCheck())
      break;

   }


   FinishSet();

}

bool Viewer::FinishCheck()
{
  unique_lock<mutex> Lock(mMutex_ViewerFinish);
  return mbRequestFinish;
}

void Viewer::FinishSet()
{
  unique_lock<mutex> Lock(mMutex_ViewerFinish);
   mbFinish = true;
}

void Viewer::SetFinishRequest()
{
  unique_lock<mutex> Lock(mMutex_ViewerFinish);
  mbRequestFinish = true;
}

bool Viewer::Finished()
{
  unique_lock<mutex> Lock(mMutex_ViewerFinish);
  return mbFinish;
}

}//end of namespace
