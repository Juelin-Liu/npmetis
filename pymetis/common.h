//
// Created by juelin on 6/17/24.
//

#ifndef CPPMETIS_COMMON_H
#define CPPMETIS_COMMON_H
#include "tcb/span.hpp"

namespace std {
    using namespace tcb;
}
namespace pymetis
{
 // mt-metis
 using id_t=uint32_t;
 using idx_t=uint64_t;
 using wgt_t=int64_t;
 
 // metis
 using metis_idx_t = int64_t;
}


#endif //CPPMETIS_COMMON_H
