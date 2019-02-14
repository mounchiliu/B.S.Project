#ifndef VOCABULARY_H
#define VOCABULARY_H


//#include "Thirdparty/DBow3/src/Vocabulary.h"
//#include "Thirdparty/DBow3/src/DescManip.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/FORB.h"

namespace ORB_SLAM
{


typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
//DBoW3::Vocabulary ORBVocabulary;


}




#endif // VOCABULARY_H
