#include "Optimization.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"





#include<Eigen/Dense>




namespace ORB_SLAM{

Optimization::Optimization()
{
}

//Only optimize pose matrix
void Optimization::Optimization_Pose(Frame *pF)
{

  //Initialize optimizer
  g2o::SparseOptimizer Optimizer;
  Optimizer.setVerbose(false);

  // solver for BA/3D SLAM
  //linear block solover (6 degrees of freedom for camera pose, 3 degrees of freedom for landmark)
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  //Use Levenberg–Marquardt algorithm
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

  Optimizer.setAlgorithm(solver);


  //Add the pose of the Current Frame as vertices
  g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap(); //SE(3) Pose
  v_se3->setId(0);
  //not set as a fixed point
  //need to be optimized
  v_se3->setFixed(false);
  //set the initial estimate
  g2o::SE3Quat T = Converter::Convert2SE3(pF->mTWorld2Cam);//Translate to suitable form
  v_se3->setEstimate(T);//For initialization

  Optimizer.addVertex(v_se3);

  //Set MapPoints as edges

  int num = pF->mNumKeypoints;

  //initialize  
  vector<pair<g2o::EdgeStereoSE3ProjectXYZOnlyPose*,size_t>> vpEdges_Index;//store Unary Edges information and index
  vpEdges_Index.reserve(num);
  vector<pair<g2o::EdgeSE3ProjectXYZOnlyPose*,size_t>> vpEdges_Index_RightNotOk;//Right not available
  vpEdges_Index_RightNotOk.reserve(num);

  {
  unique_lock<mutex> Lock(MapPoint::mMutex);
  for(int i=0;i<num;i++)
  {
    MapPoint* pMp = pF->mvMapPoint[i];
    if(pMp)
    {
      //Right not available
      if(pF->mvDepthWithKPU[i]<0)
      {
        pF->mvOutliers[i] = false;

        Eigen::Matrix<double,2,1> Obs;
        cv::KeyPoint &Kp =pF->mvUndisKP[i];
        //Obs
        Obs<<Kp.pt.x, Kp.pt.y;

        //Add Unary edges
        g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

        e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(0)));
        e->setMeasurement(Obs);

        float square = pF->mvScaleFactors[Kp.octave] * pF->mvScaleFactors[Kp.octave];
        float inv_square = 1.0f / square;
        //Information matrix//
        Eigen::Matrix2d Info_Matrix = Eigen::Matrix2d::Identity()*inv_square;
        e->setInformation(Info_Matrix);

        //add camera parameters to the edge
        e->fx = pF->fx;
        e->fy = pF->fy;
        e->cx = pF->cx;
        e->cy = pF->cy;
        cv::Mat WorldPosition = pMp->GetWorldPosition();
        e->Xw[0] = WorldPosition.at<float>(0);
        e->Xw[1] = WorldPosition.at<float>(1);
        e->Xw[2] = WorldPosition.at<float>(2);

        //use Huber loss function to enable it less sensitive to errors
        double delta = sqrt(5.991);
        g2o::RobustKernelHuber* RobustKernel = new g2o::RobustKernelHuber;
        e->setRobustKernel(RobustKernel);
        RobustKernel->setDelta(delta);
        //if error<=delta (inlier)
        //loss function = squared loss function(1/2 * (y-f(x))^2)
        //otherwise use huber loss function
        //loss function = delta * |y-f(x)| - 1/2 * delta^2

        Optimizer.addEdge(e);

        vpEdges_Index_RightNotOk.push_back(make_pair(e,i));


      }
      else
      {
        pF->mvOutliers[i] = false;

        //Set edge
        Eigen::Matrix<double,3,1> ObservationOfKP;
        cv::KeyPoint &Kp = pF->mvUndisKP[i];
        float &URKP = pF->mvDepthWithKPU[i];

        ObservationOfKP << Kp.pt.x, Kp.pt.y, URKP;


        //Add Unary edges

        g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

        e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(0)));
        e->setMeasurement(ObservationOfKP);

        float square = pF->mvScaleFactors[Kp.octave] * pF->mvScaleFactors[Kp.octave];
        float inv_square = 1.0f / square;
        //Information matrix
        Eigen::Matrix3d Info_Matrix = Eigen::Matrix3d::Identity()*inv_square;
        e->setInformation(Info_Matrix);

        //add camera parameters to the edge
        e->fx = pF->fx;
        e->fy = pF->fy;
        e->cx = pF->cx;
        e->cy = pF->cy;
        e->bf = pF->mbfx;
        cv::Mat WorldPosition = pMp->GetWorldPosition();
        e->Xw[0] = WorldPosition.at<float>(0);
        e->Xw[1] = WorldPosition.at<float>(1);
        e->Xw[2] = WorldPosition.at<float>(2);

        //use Huber loss function to enable it less sensitive to errors
        double delta = sqrt(7.815);
        g2o::RobustKernelHuber* RobustKernel = new g2o::RobustKernelHuber;
        e->setRobustKernel(RobustKernel);
        RobustKernel->setDelta(delta);
        //if error<=delta (inlier)
        //loss function = squared loss function(1/2 * (y-f(x))^2)
        //otherwise use huber loss function
        //loss function = delta * |y-f(x)| - 1/2 * delta^2

        Optimizer.addEdge(e);

        vpEdges_Index.push_back(make_pair(e,i));

      }
    }
  }
  }
  int numBad;
  //perform optimization 4 iterations in which has 4 times optimization
  for(size_t it = 0; it<4;it++)
  {
    numBad = 0;

    v_se3->setEstimate(Converter::Convert2SE3(pF->mTWorld2Cam));
    //Optimiza the edge with level 0
    Optimizer.initializeOptimization(0);//Initialize Optimization
    Optimizer.optimize(10);

    for(size_t i = 0; i<vpEdges_Index.size();i++)
    {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdges_Index[i].first;
      size_t index = vpEdges_Index[i].second;

      if(pF->mvOutliers[index])
        e->computeError();

      //If the edge has higher chi-square, this edge is an outlier
      float chi_squared = e->chi2(); //smaller chi2 means less error
      if(chi_squared>7.815)
      {
        pF->mvOutliers[index] = true;
        e->setLevel(1);//Do not Optimize it
        numBad++;
      }
      else
      {
        e->setLevel(0);//Optimize it
        pF->mvOutliers[index]=false;
      }

      if(it==2)
        e->setRobustKernel(nullptr);
    }

    for(size_t i = 0; i<vpEdges_Index_RightNotOk.size();i++)
    {
      g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdges_Index_RightNotOk[i].first;
      size_t index = vpEdges_Index_RightNotOk[i].second;

      if(pF->mvOutliers[index])
        e->computeError();

      //If the edge has higher chi-square, this edge is an outlier
      float chi_squared = e->chi2(); //smaller chi2 means less error
      if(chi_squared>5.991)//buxiangguan
      {
        pF->mvOutliers[index] = true;
        e->setLevel(1);//Do not Optimize it
        numBad++;
      }
      else
      {
        e->setLevel(0);//Optimize it
        pF->mvOutliers[index]=false;
      }

      if(it==2)
        e->setRobustKernel(nullptr);
    }

  if(Optimizer.edges().size()<10)
    break;
  }



  //recover optimized pose
  g2o::VertexSE3Expmap* v_se3_recover = static_cast<g2o::VertexSE3Expmap*>(Optimizer.vertex(0));
  g2o::SE3Quat se3_recover = v_se3_recover->estimate();
  cv::Mat cameraPose = Converter::toCVMatrix(se3_recover);

  pF->UpdatePose(cameraPose);


}

//Local Optimization
void Optimization::LocalOptimization(KeyFrame *pKF, Map *pMap, bool* bStop)
{
  list<KeyFrame*> KFsInLocalMap;

  //1. add the Current KF to the Map
  KFsInLocalMap.push_back(pKF);
  pKF->mnLocalBAKFid = pKF->mnID;

  vector<KeyFrame*> vpConnectedKFs = pKF->GetConnectedKFs();//ALL CONNECTED KFS
  for(int i=0; i<vpConnectedKFs.size(); i++)
  {
    KeyFrame* _pKF = vpConnectedKFs[i];
    _pKF->mnLocalBAKFid = pKF->mnID;
    if(!_pKF->IsBad())
      KFsInLocalMap.push_back(_pKF);
  }

  //Get all MPs in all Connected KFs
  list<MapPoint*> MPsInLocalMap;
  for(list<KeyFrame*>::iterator it=KFsInLocalMap.begin(), end = KFsInLocalMap.end(); it!=end; it++)
  {
    vector<MapPoint*> vpMPs = (*it)->GetAllMapPoints();
    //Go through each MPs
    for(vector<MapPoint*>::iterator itMP = vpMPs.begin(), endMP = vpMPs.end(); itMP!=endMP; itMP++)
    {
      MapPoint* pMP = *itMP;
      if(!pMP)
        continue;
      if(pMP->BadMP())
        continue;
      if(pMP->mnLocalBAKFid==pKF->mnID)
        continue;
      MPsInLocalMap.push_back(pMP);
      pMP->mnLocalBAKFid = pKF->mnID;
    }
  }

  //For all MPs, find KFs can observe the MP but not connected to pKF
  //Set them as fixed points
  list<KeyFrame*> lpFixedKFs;
  for(list<MapPoint*>::iterator it=MPsInLocalMap.begin(), end=MPsInLocalMap.end(); it!=end; it++)
  {
    //Get observations
    map<KeyFrame*,size_t> Obs = (*it)->GetObsInfo();
    for(map<KeyFrame*,size_t>::iterator itObs=Obs.begin(), endObs=Obs.end(); itObs!=endObs; itObs++)
    {
      KeyFrame* pKF_Obs = itObs->first;
      //If this KF is not connected to pKF
      if(pKF_Obs->mnLocalBAKFid!=pKF->mnID && pKF_Obs->mnLocalBAFixedKF != pKF->mnID)
      {
        pKF_Obs ->mnLocalBAFixedKF = pKF->mnID;
        if(!pKF_Obs->IsBad())
          lpFixedKFs.push_back(pKF_Obs);

      }
    }
  }

  //Set up optimizer
  g2o::SparseOptimizer Optimizer;
  Optimizer.setVerbose(false);

  // solver for BA/3D SLAM
  //linear block solover (6 degrees of freedom for camera pose, 3 degrees of freedom for landmark)
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  //Use Levenberg–Marquardt algorithm
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

  Optimizer.setAlgorithm(solver);


  //Find maxID of KF
  unsigned long maxKF_ID=0;
  //Stop optimization?
  if(bStop)
    Optimizer.setForceStopFlag(bStop);


  //Add the pose of each  KeyFrame as vertices
  for(list<KeyFrame*>::iterator itKF=KFsInLocalMap.begin(), endKF=KFsInLocalMap.end(); itKF!=endKF; itKF++)
  {
    KeyFrame* pKFi = *itKF;
    g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap(); //SE(3) Pose
    v_se3->setId(pKFi->mnID);
    //only set the first one as fixed value
    v_se3->setFixed(pKFi->mnID==0);
    //set the initial estimate
    g2o::SE3Quat T = Converter::Convert2SE3(pKFi->GetPose());//Translate to suitable form
    v_se3->setEstimate(T);//For initialization

    Optimizer.addVertex(v_se3);

    if(pKFi->mnID>maxKF_ID)
      maxKF_ID = pKFi->mnID;
  }

  //Set fixed KFs
  for(list<KeyFrame*>::iterator itFixed = lpFixedKFs.begin(), endFixed=lpFixedKFs.end(); itFixed!=endFixed; itFixed++)
  {
    KeyFrame* pKFi = *itFixed;
    g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap(); //SE(3) Pose
    v_se3->setId(pKFi->mnID);
    //set as fixed value
    v_se3->setFixed(true);
    //set the initial estimate
    g2o::SE3Quat T = Converter::Convert2SE3(pKFi->GetPose());//Translate to suitable form
    v_se3->setEstimate(T);//For initialization

    Optimizer.addVertex(v_se3);

    if(pKFi->mnID>maxKF_ID)
      maxKF_ID = pKFi->mnID;
  }

  //Set MPs vertices
  int n = KFsInLocalMap.size()+MPsInLocalMap.size()+lpFixedKFs.size();
  vector<KeyFrame*> vpEdgeKF;
  vpEdgeKF.reserve(n);

  vector<KeyFrame*> vpEdgeKF_rightNotOk;
  vpEdgeKF_rightNotOk.reserve(n);

  vector<MapPoint*> vpEdgeMP;
  vpEdgeMP.reserve(n);

  vector<MapPoint*> vpEdgeMP_rightNotOk;
  vpEdgeMP_rightNotOk.reserve(n);

  vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdges;
  vpEdges.reserve(n);

  vector<g2o::EdgeSE3ProjectXYZ*> vpEdges_rightNotOk;
  vpEdges_rightNotOk.reserve(n);

  for(list<MapPoint*>::iterator itMP=MPsInLocalMap.begin(), endMP=MPsInLocalMap.end(); itMP!=endMP; itMP++)
  {
    MapPoint* pMP = *itMP;
    g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    point->setEstimate(Converter::toEigenMatrix(pMP->mWorldPos));
    int id = pMP->mnID_MP+maxKF_ID+1;
    point->setId(id);
    //this node should be marginalized out in the optimization
    point->setMarginalized(true);
    Optimizer.addVertex(point);

    //Get the Obs of the MP
    map<KeyFrame*,size_t> Obs = pMP->GetObsInfo();

    //set edges
    for(map<KeyFrame*,size_t>::iterator itObs=Obs.begin(), endObs=Obs.end(); itObs!=endObs; itObs++)
    {
      KeyFrame* pKFi = itObs->first;
      if(!pKFi->IsBad())
      {
        const cv::KeyPoint &kp = pKFi->mvUndisKP[itObs->second];

        if(pKFi->mvDepthWithKP[itObs->second]<0)
        {
          Eigen::Matrix<double,2,1> Obs;
          Obs<<kp.pt.x,kp.pt.y;

          //Add edges
          g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

          e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(id)));
          e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(pKFi->mnID)));


          e->setMeasurement(Obs);

          float square = pKFi->mvScaleFactor[kp.octave] * pKFi->mvScaleFactor[kp.octave];
          float inv_square = 1.0f / square;
          //Information matrix
          Eigen::Matrix2d Info_Matrix = Eigen::Matrix2d::Identity()*inv_square;
          e->setInformation(Info_Matrix);

          //add camera parameters to the edge
          e->fx = pKFi->mfx;
          e->fy = pKFi->mfy;
          e->cx = pKFi->mcx;
          e->cy = pKFi->mcy;

          //use Huber loss function to enable it less sensitive to errors
          double delta = sqrt(5.991);
          g2o::RobustKernelHuber* RobustKernel = new g2o::RobustKernelHuber;
          e->setRobustKernel(RobustKernel);
          RobustKernel->setDelta(delta);
          //if error<=delta (inlier)
          //loss function = squared loss function(1/2 * (y-f(x))^2)
          //otherwise use huber loss function
          //loss function = delta * |y-f(x)| - 1/2 * delta^2

          Optimizer.addEdge(e);

          vpEdges_rightNotOk.push_back(e);
          vpEdgeKF_rightNotOk.push_back(pKFi);
          vpEdgeMP_rightNotOk.push_back(pMP);

        }else
        {
          Eigen::Matrix<double,3,1> Obs;
          const float kp_r = pKFi->mvDepthWithKP[itObs->second];
          Obs<<kp.pt.x,kp.pt.y,kp_r;

          //Add edges
          g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

          e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(id)));
          e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(Optimizer.vertex(pKFi->mnID)));


          e->setMeasurement(Obs);

          float square = pKFi->mvScaleFactor[kp.octave] * pKFi->mvScaleFactor[kp.octave];
          float inv_square = 1.0f / square;
          //Information matrix
          Eigen::Matrix3d Info_Matrix = Eigen::Matrix3d::Identity()*inv_square;
          e->setInformation(Info_Matrix);

          //add camera parameters to the edge
          e->fx = pKFi->mfx;
          e->fy = pKFi->mfy;
          e->cx = pKFi->mcx;
          e->cy = pKFi->mcy;
          e->bf = pKFi->mbfx;

          //use Huber loss function to enable it less sensitive to errors
          double delta = sqrt(7.815);
          g2o::RobustKernelHuber* RobustKernel = new g2o::RobustKernelHuber;
          e->setRobustKernel(RobustKernel);
          RobustKernel->setDelta(delta);
          //if error<=delta (inlier)
          //loss function = squared loss function(1/2 * (y-f(x))^2)
          //otherwise use huber loss function
          //loss function = delta * |y-f(x)| - 1/2 * delta^2

          Optimizer.addEdge(e);

          vpEdges.push_back(e);
          vpEdgeKF.push_back(pKFi);
          vpEdgeMP.push_back(pMP);


        }
      }
    }
  }

  if(bStop)
    if(*bStop)
      return;

  //Start Optimization
  Optimizer.initializeOptimization();
  Optimizer.optimize(10);

  bool bContinue=true;
  if(bStop)
    if(*bStop)
      bContinue=false;

  if(bContinue)
  {
    //Detect outliers & Mark it
    //Do not Optimize it
    for(size_t i=0; i<vpEdges.size();i++)
    {
      g2o::EdgeStereoSE3ProjectXYZ* e = vpEdges[i];
      MapPoint* pMP = vpEdgeMP[i];

      if(!pMP)
        continue;
      if(pMP->BadMP())
        continue;

      //If the edge has higher chi-square, this edge is an outlier
      float chi_squared = e->chi2(); //smaller chi2 means less error
      if(chi_squared>7.815 || !e->isDepthPositive())
      {
        e->setLevel(1);//Do not optimize
      }


      e->setRobustKernel(nullptr);//Do not use the Robust Function
    }

    for(size_t i=0; i<vpEdges_rightNotOk.size();i++)
    {
      g2o::EdgeSE3ProjectXYZ* e = vpEdges_rightNotOk[i];
      MapPoint* pMP = vpEdgeMP_rightNotOk[i];

      if(!pMP)
        continue;
      if(pMP->BadMP())
        continue;

      //If the edge has higher chi-square, this edge is an outlier
      float chi_squared = e->chi2(); //smaller chi2 means less error
      if(chi_squared>5.991 || !e->isDepthPositive())
      {
        e->setLevel(1);//Do not optimize
      }


      e->setRobustKernel(nullptr);//Do not use the Robust Function

    }


    //Optimize again without outliers
    Optimizer.initializeOptimization(0);
    Optimizer.optimize(10);
  }
  vector<pair<KeyFrame*,MapPoint*>> vToBeErased;
  vToBeErased.reserve(vpEdges.size()+vpEdges_rightNotOk.size());

  //Check inliers
  for(size_t i=0; i<vpEdges.size();i++)
  {
    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdges[i];
    MapPoint* pMP = vpEdgeMP[i];

    if(!pMP)
      continue;
    if(pMP->BadMP())
      continue;

    //If the edge has higher chi-square, this edge is an outlier
    float chi_squared = e->chi2(); //smaller chi2 means less error
    if(chi_squared>7.815 || !e->isDepthPositive())
    {
      KeyFrame* pKFi = vpEdgeKF[i];
      vToBeErased.push_back(make_pair(pKFi,pMP));
    }
  }

  for(size_t i=0; i<vpEdges_rightNotOk.size();i++)
  {
    g2o::EdgeSE3ProjectXYZ* e = vpEdges_rightNotOk[i];
    MapPoint* pMP = vpEdgeMP_rightNotOk[i];

    if(!pMP)
      continue;
    if(pMP->BadMP())
      continue;

    //If the edge has higher chi-square, this edge is an outlier
    float chi_squared = e->chi2(); //smaller chi2 means less error
    if(chi_squared>5.991 || !e->isDepthPositive())
    {
      KeyFrame* pKFi = vpEdgeKF_rightNotOk[i];
      vToBeErased.push_back(make_pair(pKFi,pMP));
    }
  }

  unique_lock<mutex> lock(pMap->mMutex_MapUpdate);

  if(!vToBeErased.empty())
  {
    for(size_t i=0; i<vToBeErased.size(); i++)
    {
      //Get KF
      KeyFrame* pKFi = vToBeErased[i].first;
      //Get MP
      MapPoint* pMPi = vToBeErased[i].second;

      //Erase Info
      pKFi->DeleteMPinKF(pMPi);
      pMPi->EraseMPObservation(pKFi);
    }
  }

  //Update optimized data

  //Keyframes
  for(list<KeyFrame*>::iterator it=KFsInLocalMap.begin(), end=KFsInLocalMap.end(); it!=end; it++)
  {
      KeyFrame* pKF = *it;
      g2o::VertexSE3Expmap* v_se3_recover = static_cast<g2o::VertexSE3Expmap*>(Optimizer.vertex(pKF->mnID));
      g2o::SE3Quat se3_recover = v_se3_recover->estimate();
      cv::Mat cameraPose = Converter::toCVMatrix(se3_recover);

      pKF->UpdatePose(cameraPose);
  }

  //MapPoints
  for(list<MapPoint*>::iterator it=MPsInLocalMap.begin(), end=MPsInLocalMap.end(); it!=end; it++)
  {
      MapPoint* pMP = *it;
      g2o::VertexSBAPointXYZ* v_Point = static_cast<g2o::VertexSBAPointXYZ*>(Optimizer.vertex(pMP->mnID_MP+maxKF_ID+1));
      cv::Mat Position = Converter::toCVMatrix(v_Point->estimate());
      pMP->SetWorldPosition(Position);
      pMP->UpdateViewDirAndScaleInfo();
  }
}


g2o::SE3Quat Converter::Convert2SE3(const cv::Mat &T)
{
  Eigen::Matrix<double,3,3> R; //Rotation Matrix (3X3)

  R<< T.at<float>(0,0), T.at<float>(0,1), T.at<float>(0,2),
      T.at<float>(1,0), T.at<float>(1,1), T.at<float>(1,2),
      T.at<float>(2,0), T.at<float>(2,1), T.at<float>(2,2);

  //Translation Matrix
  Eigen::Matrix<double,3,1> t;
  t<< T.at<float>(0,3), T.at<float>(1,3), T.at<float>(2,3);

  return g2o::SE3Quat(R,t);
}

cv::Mat Converter::toCVMatrix(const g2o::SE3Quat &se3)
{
  //For transformation Matrix, size = 4x4
  Eigen::Matrix<double,4,4> Matrix_Eigen = se3.to_homogeneous_matrix();
  cv::Mat Matrix_CV(4,4,CV_32F);
  for(int i=0;i<4;i++)
      for(int j=0;j<4;j++)
        Matrix_CV.at<float>(i,j)=Matrix_Eigen(i,j);

  return Matrix_CV.clone();
}

cv::Mat Converter::toCVMatrix(const Eigen::Matrix<double,3,1> &em)
{
  cv::Mat result(3,1,CV_32F);
  for(int i=0;i<3;i++)
      result.at<float>(i)=em(i);

  return result.clone();
}


Eigen::Matrix<double,3,1> Converter::toEigenMatrix(const cv::Mat &mat)
{
  Eigen::Matrix<double,3,1> e_matrix;
  e_matrix<< mat.at<float>(0), mat.at<float>(1), mat.at<float>(2);

  return e_matrix;
}

Eigen::Matrix<double,3,3> Converter::toEigenMatrix_3_3(const cv::Mat &mat)
{
  Eigen::Matrix<double,3,3> e_matrix;
  e_matrix<< mat.at<float>(0,0), mat.at<float>(0,1), mat.at<float>(0,2),
             mat.at<float>(1,0), mat.at<float>(1,1), mat.at<float>(1,2),
             mat.at<float>(2,0), mat.at<float>(2,1), mat.at<float>(2,2);

  return e_matrix;
}


vector<float> Converter::toQuat(cv::Mat &mat)
{
  Eigen::Matrix<double,3,3> e_matrix = toEigenMatrix_3_3(mat);
  Eigen::Quaterniond quat(e_matrix);

  vector<float> vec(4);

  vec[0] = quat.x();
  vec[1] = quat.y();
  vec[2] = quat.z();
  vec[3] = quat.w();

  return vec;
}
}
