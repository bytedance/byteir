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

#ifndef _PIMRANK_H_
#define _PIMRANK_H_

#include <vector>

#include "AddressMapping.h"
#include "BusPacket.h"
#include "Configuration.h"
#include "PIMBlock.h"
#include "PIMCmd.h"
#include "Rank.h"
#include "SimulatorObject.h"

using namespace std;
using namespace DRAMSim;

namespace DRAMSim
{
#define OUTLOG_ALL(msg)                                                                                          \
    msg << " ch" << getChanId() << " ra" << getRankId() << " bg" << config.addrMapping.bankgroupId(packet->bank) \
        << " b" << packet->bank << " r" << packet->row << " c" << packet->column << " @" << currentClockCycle
#define OUTLOG_CH_RA(msg) msg << " ch" << getChanId() << " ra" << getRankId() << " @" << currentClockCycle
#define OUTLOG_PRECHARGE(msg)                                                                                    \
    msg << " ch" << getChanId() << " ra" << getRankId() << " bg" << config.addrMapping.bankgroupId(packet->bank) \
        << " b" << packet->bank << " r" << bankStates[packet->bank].openRowAddress << " @" << currentClockCycle
#define OUTLOG_GRF_A(msg)                                                                                              \
    msg << " ch" << getChanId() << " ra" << getRankId() << " pb" << packet->bank / 2 << " reg" << packet->column - 0x8 \
        << " @" << currentClockCycle
#define OUTLOG_GRF_B(msg)                                                                      \
    msg << " ch" << getChanId() << " ra" << getRankId() << " pb" << packet->bank / 2 << " reg" \
        << packet->column - 0x18 << " @" << currentClockCycle
#define OUTLOG_B_GRF_A(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " reg" << packet->column - 0x8 << " @" << currentClockCycle
#define OUTLOG_B_GRF_B(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " reg" << packet->column - 0x18 << " @" << currentClockCycle
#define OUTLOG_B_CRF(msg) \
    msg << " ch" << getChanId() << " ra" << getRankId() << " idx" << packet->column - 0x4 << " @" << currentClockCycle

class Rank;  // forward declaration

class PIMRank : public SimulatorObject
{
   private:
    int chanId;
    int rankId;
    ostream& dramsimLog;
    Configuration& config;
    int pimPC_, lastJumpIdx_, numJumpToBeTaken_, lastRepeatIdx_, numRepeatToBeDone_;
    bool pimOpMode_, toggleEvenBank_, toggleOddBank_, toggleRa12h_, useAllGrf_, crfExit_;

   public:
    PIMRank(ostream& simLog, Configuration& configuration);
    ~PIMRank() {}

    void attachRank(Rank* r);
    int getChanId() const;
    void setChanId(int id);
    int getRankId() const;
    void setRankId(int id);
    void update();
    void readHab(BusPacket* packet);
    void writeHab(BusPacket* packet);
    void doPIM(BusPacket* packet);
    void doPIMBlock(BusPacket* packet, PIMCmd curCmd, int pimblock_id);
    void controlPIM(BusPacket* packet);
    void readOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx, bool is_auto, bool is_mac);
    void writeOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx, bool is_auto, bool is_mac);
    bool isToggleCond(BusPacket* packet);

    union crf_t {
        uint32_t data[32];
        BurstType bst[4];
        crf_t() { memset(data, 0, sizeof(uint32_t) * 32); }
    } crf;

    unsigned inline getGrfIdx(unsigned idx) { return idx & 0x7; }
    unsigned inline getGrfIdxHigh(unsigned r, unsigned c) { return ((r & 0x1) << 2 | ((c >> 3) & 0x3)); }

    Rank* rank;
    vector<PIMBlock> pimBlocks;
};
}  // namespace DRAMSim
#endif
