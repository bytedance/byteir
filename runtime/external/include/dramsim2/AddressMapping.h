/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *notice,
 *        this list of conditions and the following disclaimer in the
 *documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/
#ifndef ADDRESS_MAPPING_H
#define ADDRESS_MAPPING_H

#include <cstdint>

#include "SystemConfiguration.h"

namespace DRAMSim
{
class AddrMapping
{
   public:
    AddrMapping();
    void addressMapping(uint64_t physicalAddress, unsigned& channel, unsigned& rank, unsigned& bank, unsigned& row,
                        unsigned& col);

    uint64_t inline diffBitWidth(uint64_t* physicalAddress, uint64_t BitWidth)
    {
        uint64_t tempA = *physicalAddress;
        *physicalAddress = *physicalAddress >> BitWidth;
        uint64_t tempB = *physicalAddress << BitWidth;
        return tempA ^ tempB;
    }

    unsigned bankgroupId(int bank);
    bool isSameBankgroup(int bank0, int bank1);

   private:
    uint64_t transactionSize;
    uint64_t transactionMask;
    uint64_t channelBitWidth;
    uint64_t rankBitWidth;
    uint64_t bankBitWidth;
    uint64_t bankgroupBitWidth;
    uint64_t rowBitWidth;
    uint64_t colBitWidth;
    uint64_t byteOffsetWidth;
    uint64_t colLowBitWidth;
    uint64_t colHighBitWidth;

    unsigned num_chans_;
    int num_bank_per_bg_;
    AddressMappingScheme addressMappingScheme;
};
}  // namespace DRAMSim

#endif
