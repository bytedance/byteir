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

#ifndef UTILS_H_
#define UTILS_H_

namespace DRAMSim
{
#define Byte2GB(x) ((x) >> 30)
#define Byte2MB(x) ((x) >> 20)
#define Byte2KB(x) ((x) >> 10)

inline unsigned uLog2(unsigned value)
{
    unsigned logBase2 = 0;
    unsigned orig = value;
    value >>= 1;
    while (value > 0) {
        value >>= 1;
        logBase2++;
    }
    if ((unsigned)1 << logBase2 < orig) logBase2++;
    return logBase2;
}

inline bool isPowerOfTwo(unsigned long x) { return (1UL << uLog2(x)) == x; }

};  // namespace DRAMSim

#endif  // UTILS_H
