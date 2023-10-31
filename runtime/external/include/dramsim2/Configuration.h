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

#ifndef CONFIGURATION_CACHE_H_
#define CONFIGURATION_CACHE_H_

#include <algorithm>
#include <string>

#include "AddressMapping.h"
#include "SystemConfiguration.h"

namespace DRAMSim
{
class Configuration
{
   public:
    Configuration(AddrMapping& am) : addrMapping(am)
    {
        AL = getConfigParam(UINT, "AL");
        BL = getConfigParam(UINT, "BL");
        CMD_QUEUE_DEPTH = getConfigParam(UINT, "CMD_QUEUE_DEPTH");
        DEVICE_WIDTH = getConfigParam(UINT, "DEVICE_WIDTH");
        EPOCH_LENGTH = getConfigParam(UINT, "EPOCH_LENGTH");
        HISTOGRAM_BIN_SIZE = getConfigParam(UINT, "HISTOGRAM_BIN_SIZE");
        JEDEC_DATA_BUS_BITS = getConfigParam(UINT, "JEDEC_DATA_BUS_BITS");
        NUM_BANKS = getConfigParam(UINT, "NUM_BANKS");
        NUM_COLS = getConfigParam(UINT, "NUM_COLS");
        NUM_CHANS = getConfigParam(UINT, "NUM_CHANS");
        NUM_PIM_BLOCKS = getConfigParam(UINT, "NUM_PIM_BLOCKS");
        NUM_RANKS = getConfigParam(UINT, "NUM_RANKS");
        NUM_ROWS = getConfigParam(UINT, "NUM_ROWS");
        RL = getConfigParam(UINT, "RL");
        tCCDL = getConfigParam(UINT, "tCCDL");
        tCCDS = getConfigParam(UINT, "tCCDS");
        tCK = getConfigParam(FLOAT, "tCK");
        tCMD = getConfigParam(UINT, "tCMD");
        tCKE = getConfigParam(UINT, "tCKE");
        tRAS = getConfigParam(UINT, "tRAS");
        tRC = getConfigParam(UINT, "tRC");
        tRCDRD = getConfigParam(UINT, "tRCDRD");
        tRCDWR = getConfigParam(UINT, "tRCDWR");
        tREFI = getConfigParam(UINT, "tREFI");
        tREFISB = getConfigParam(UINT, "tREFISB");
        tRFC = getConfigParam(UINT, "tRFC");
        tRP = getConfigParam(UINT, "tRP");
        tRRDL = getConfigParam(UINT, "tRRDL");
        tRRDS = getConfigParam(UINT, "tRRDS");
        tRTP = getConfigParam(UINT, "tRTP");
        tRTPL = getConfigParam(UINT, "tRTPL");
        tRTPS = getConfigParam(UINT, "tRTPS");
        tRTRS = getConfigParam(UINT, "tRTRS");
        tWR = getConfigParam(UINT, "tWR");
        tWTRL = getConfigParam(UINT, "tWTRL");
        tWTRS = getConfigParam(UINT, "tWTRS");
        tXP = getConfigParam(UINT, "tXP");
        TOTAL_ROW_ACCESSES = getConfigParam(UINT, "TOTAL_ROW_ACCESSES");
        TRANS_QUEUE_DEPTH = getConfigParam(UINT, "TRANS_QUEUE_DEPTH");
        WL = getConfigParam(UINT, "WL");
        XAW = getConfigParam(UINT, "XAW");

        PIM_MODE = PIMConfiguration::getPIMMode();
        PIM_PRECISION = PIMConfiguration::getPIMPrecision();
        ROW_BUFFER_POLICY = PIMConfiguration::getRowBufferPolicy();
        SCHEDULING_POLICY = PIMConfiguration::getSchedulingPolicy();
        QUEUING_STRUCTURE = PIMConfiguration::getQueueingStructure();
        ADDRESS_MAPPING_SCHEME = PIMConfiguration::getAddressMappingScheme();

        READ_TO_PRE_DELAY = (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL);
        READ_TO_PRE_DELAY_LONG = (AL + BL / 2 + max(tRTPL, tCCDL) - tCCDL);
        READ_TO_PRE_DELAY_SHORT = (AL + BL / 2 + max(tRTPS, tCCDS) - tCCDS);
        WRITE_TO_PRE_DELAY = (WL + BL / 2 + tWR);
        READ_TO_WRITE_DELAY = (RL + BL / 2 + tRTRS - WL);
        READ_AUTOPRE_DELAY = (AL + tRTP + tRP);
        WRITE_AUTOPRE_DELAY = (WL + BL / 2 + tWR + tRP);
        WRITE_TO_READ_DELAY_B_LONG = (WL + BL / 2 + tWTRL);   // interbank
        WRITE_TO_READ_DELAY_B_SHORT = (WL + BL / 2 + tWTRS);  // interbank
        WRITE_TO_READ_DELAY_R = max((int)(WL + BL / 2 + tRTRS) - (int)RL, (int)0);

        if (NUM_CHANS == 0) {
            throw invalid_argument("Not allowed zero channel");
        }

        setDebugConfiguration();
        setOutputConfiguration();
    }

    void setDebugConfiguration()
    {
        DEBUG_PIM_BLOCK = getConfigParam(BOOL, "DEBUG_PIM_BLOCK");
        DEBUG_TRANS_Q = getConfigParam(BOOL, "DEBUG_TRANS_Q");
        DEBUG_CMD_Q = getConfigParam(BOOL, "DEBUG_CMD_Q");
        DEBUG_ADDR_MAP = getConfigParam(BOOL, "DEBUG_ADDR_MAP");
        DEBUG_BANKSTATE = getConfigParam(BOOL, "DEBUG_BANKSTATE");
        DEBUG_BUS = getConfigParam(BOOL, "DEBUG_BUS");
        DEBUG_BANKS = getConfigParam(BOOL, "DEBUG_BANKS");
        DEBUG_POWER = getConfigParam(BOOL, "DEBUG_POWER");
        DEBUG_CMD_TRACE = getConfigParam(BOOL, "DEBUG_CMD_TRACE");
        DEBUG_PIM_TIME = getConfigParam(BOOL, "DEBUG_PIM_TIME");
    }

    void setOutputConfiguration()
    {
        PRINT_CHAN_STAT = getConfigParam(BOOL, "PRINT_CHAN_STAT");
        VIS_FILE_OUTPUT = getConfigParam(BOOL, "VIS_FILE_OUTPUT");
        VERIFICATION_OUTPUT = getConfigParam(BOOL, "VERIFICATION_OUTPUT");
        SHOW_SIM_OUTPUT = getConfigParam(BOOL, "SHOW_SIM_OUTPUT");
        LOG_OUTPUT = getConfigParam(BOOL, "LOG_OUTPUT");
        SIM_TRACE_FILE = getConfigParam(STRING, "SIM_TRACE_FILE");
    }

    unsigned AL;
    unsigned BL;
    unsigned CMD_QUEUE_DEPTH;
    unsigned DEVICE_WIDTH;
    unsigned EPOCH_LENGTH;
    unsigned HISTOGRAM_BIN_SIZE;
    unsigned JEDEC_DATA_BUS_BITS;
    unsigned NUM_BANKS;
    unsigned NUM_COLS;
    unsigned NUM_CHANS;
    unsigned NUM_PIM_BLOCKS;
    unsigned NUM_RANKS;
    unsigned NUM_ROWS;
    unsigned RL;
    unsigned tCCDL;
    unsigned tCCDS;
    float tCK;
    unsigned tCMD;
    unsigned tCKE;
    unsigned tRAS;
    unsigned tRC;
    unsigned tRCDRD;
    unsigned tRCDWR;
    unsigned tREFI;
    unsigned tREFISB;
    unsigned tRFC;
    unsigned tRP;
    unsigned tRRDL;
    unsigned tRRDS;
    unsigned tRTP;
    unsigned tRTPL;
    unsigned tRTPS;
    unsigned tRTRS;
    unsigned tWR;
    unsigned tWTRL;
    unsigned tWTRS;
    unsigned tXP;
    unsigned TOTAL_ROW_ACCESSES;
    unsigned TRANS_QUEUE_DEPTH;
    unsigned WL;
    unsigned XAW;

    PIMMode PIM_MODE;
    PIMPrecision PIM_PRECISION;
    RowBufferPolicy ROW_BUFFER_POLICY;
    SchedulingPolicy SCHEDULING_POLICY;
    QueuingStructure QUEUING_STRUCTURE;
    AddressMappingScheme ADDRESS_MAPPING_SCHEME;

    unsigned READ_TO_PRE_DELAY;
    unsigned READ_TO_PRE_DELAY_LONG;
    unsigned READ_TO_PRE_DELAY_SHORT;
    unsigned WRITE_TO_PRE_DELAY;
    unsigned READ_TO_WRITE_DELAY;
    unsigned READ_AUTOPRE_DELAY;
    unsigned WRITE_AUTOPRE_DELAY;
    unsigned WRITE_TO_READ_DELAY_B_LONG;
    unsigned WRITE_TO_READ_DELAY_B_SHORT;
    unsigned WRITE_TO_READ_DELAY_R;

    AddrMapping& addrMapping;
};

};  // namespace DRAMSim

#endif /* CONFIGURATION_CACHE_H_ */
