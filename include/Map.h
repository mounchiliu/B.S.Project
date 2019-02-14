#ifndef MAP_H
#define MAP_H




#include "MapPoint.h"
#include "KeyFrame.h"

#include <set>
#include <mutex>


using namespace std;

namespace ORB_SLAM
{
class KeyFrame;
class MapPoint;

class Map
{
public:
  Map();

  //Add KF in Map
  void AddKFInMap(KeyFrame* pKF);
  //ADD MP in Map (In MAP, dont need association with KP_ID -> THe order is not neccessary)
  void AddMPInMap(MapPoint* pMP);
  //DELETE MPs
  void DeleteMPs(MapPoint *pMP);
  //Delete KFs
  void DeleteKFs(KeyFrame *pKF);
  //Set ref MPs For Drawing Map
  void SetRefMPs(vector<MapPoint*> &MPs);
  //Get Ref MPs
  vector<MapPoint*> GetRefMPs();
  //GET num of MP
  size_t GetNumOfMP();
  //GET num of KF
  size_t GetNumOfKF();
  //GET all MPs
  vector<MapPoint*> GetMPs();
  //GET all KFs
  vector<KeyFrame*> GetKFs();

  //Lock
  mutex mMutex_CreatePoint;
  mutex mMutex_MapUpdate;

protected:


  //A SET for KeyFrames In Map
  set<KeyFrame*> mKeyFrames;
  //A SET for MPs In Map
  set<MapPoint*> mMapPoints;
  //Ref MPs for Drawing
  vector<MapPoint*> mvpRefMPsForDrawing;

  //Lock
  mutex mMutex_Map;

};

}//end of namespace


#endif // MAP_H
