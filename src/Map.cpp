#include "Map.h"

namespace ORB_SLAM
{
Map::Map()
{

}

void Map::AddKFInMap(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Map);
  mKeyFrames.insert(pKF);
}

void Map::AddMPInMap(MapPoint* pMP)
{
  unique_lock<mutex> Lock(mMutex_Map);
  mMapPoints.insert(pMP);
}

void Map::DeleteMPs(MapPoint *pMP)
{
  unique_lock<mutex> Lock(mMutex_Map);
  mMapPoints.erase(pMP);
}

void Map::DeleteKFs(KeyFrame *pKF)
{
  unique_lock<mutex> Lock(mMutex_Map);
  mKeyFrames.erase(pKF);
}

void Map::SetRefMPs(vector<MapPoint*> &MPs)
{
  unique_lock<mutex> Lock(mMutex_Map);
  mvpRefMPsForDrawing = MPs;
}

vector<MapPoint*> Map::GetRefMPs()
{
  unique_lock<mutex> Lock(mMutex_Map);
  return mvpRefMPsForDrawing;
}

size_t Map::GetNumOfMP()
{
  unique_lock<mutex> Lock(mMutex_Map);
  return mMapPoints.size();
}

size_t Map::GetNumOfKF()
{
  unique_lock<mutex> Lock(mMutex_Map);
  return mKeyFrames.size();
}

vector<MapPoint*> Map::GetMPs()
{
  unique_lock<mutex> Lock(mMutex_Map);
  return vector<MapPoint*> (mMapPoints.begin(),mMapPoints.end());
}

vector<KeyFrame*> Map::GetKFs()
{
  unique_lock<mutex> Lock(mMutex_Map);
  return vector<KeyFrame*> (mKeyFrames.begin(), mKeyFrames.end());
}



}//end of namespace
