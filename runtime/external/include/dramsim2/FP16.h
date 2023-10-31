/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef __HALF__HPP__
#define __HALF__HPP__

#include <cstdint>
#include <cstring>
#include <iostream>

#include "half.h"

using namespace std;

typedef half_float::half fp16;
float convertH2F(fp16 val);
fp16 convertF2H(float val);

union fp16i {
    fp16 fval;
    uint16_t ival;

    fp16i() { ival = 0; }
    fp16i(fp16 x) { fval = x; }
    fp16i(uint16_t x) { ival = x; }
};

bool fp16Equal(fp16 A, fp16 B, int maxUlpsDiff, float maxFsdiff);

#endif
