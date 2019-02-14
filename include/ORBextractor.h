#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

using namespace std;


namespace ORB_SLAM
{
//-----------------------------------------------------------------------------------------------
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    list<ExtractorNode>::iterator lit;
    bool bNoMore;
};
//----------------------------------------------------------------------------------------------
class ORBextractor
{
public:

    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
      vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    vector<float> inline GetScaleFactors(){
        return mvScaleFactors;
    }

    vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactors;
    }


    vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPoints(vector<vector<cv::KeyPoint> >& allKeypoints);
    vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);


    vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    vector<int> mnFeaturesPerLevel;

    vector<int> umax;

    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
};

} //namespace ORB_SLAM

#endif
