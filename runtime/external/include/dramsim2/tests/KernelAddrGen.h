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

#ifndef __ADDRGEN_HPP__
#define __ADDRGEN_HPP__

#include <sstream>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "Utils.h"

using namespace DRAMSim;

class PIMAddrManager
{
   public:
    unsigned num_chans_;
    unsigned num_ranks_;
    unsigned num_bank_groups_;
    unsigned num_banks_;
    unsigned num_rows_;
    unsigned num_cols_;
    unsigned num_pim_chans_;
    unsigned num_pim_ranks_;
    unsigned num_cols_per_bl_;

    uint64_t addrGen(unsigned chan, unsigned rank, unsigned bankgroup, unsigned bank, unsigned row, unsigned col);
    uint64_t addrGenSafe(unsigned chan, unsigned rank, unsigned bankgroup, unsigned bank, unsigned& row, unsigned& col);
    unsigned maskByBit(unsigned value, int startingBit, int endBit);

    PIMAddrManager(int num_pim_chans, int num_pim_ranks) : num_pim_chans_(num_pim_chans), num_pim_ranks_(num_pim_ranks)
    {
        num_chans_ = getConfigParam(UINT, "NUM_CHANS");
        num_ranks_ = getConfigParam(UINT, "NUM_RANKS");
        num_bank_groups_ = getConfigParam(UINT, "NUM_BANK_GROUPS");
        num_banks_ = getConfigParam(UINT, "NUM_BANKS");
        num_rows_ = getConfigParam(UINT, "NUM_ROWS");
        num_cols_ = getConfigParam(UINT, "NUM_COLS");

        num_bankgroup_bits_ = uLog2(num_bank_groups_);
        num_bank_bits_ = uLog2(num_banks_) - uLog2(num_bank_groups_);
        num_row_bits_ = uLog2(num_rows_);
        num_chan_bits_ = uLog2(num_chans_);
        num_col_bits_ = uLog2(num_cols_ / getConfigParam(UINT, "BL"));
        num_rank_bits_ = uLog2(num_ranks_);
        num_offset_bits_ = uLog2(getConfigParam(UINT, "BL") * getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8);

        num_col_low_bits_ = 2;
        num_col_high_bits_ = num_col_bits_ - num_col_low_bits_;
        /* FIXME: need to change at launch shcha */
        num_bank_low_bits_ = num_bank_bits_ / 2;
        num_bank_high_bits_ = num_bank_bits_ - num_bank_low_bits_;
        num_cols_per_bl_ = num_cols_ / getConfigParam(UINT, "BL");

        address_mapping_scheme_ = PIMConfiguration::getAddressMappingScheme();
    }

   private:
    int num_chan_bits_;
    int num_rank_bits_;
    int num_col_bits_;
    int num_row_bits_;
    int num_bank_bits_;
    int num_bankgroup_bits_;
    int num_offset_bits_;

    int num_col_low_bits_;
    int num_col_high_bits_;
    int num_bank_low_bits_;
    int num_bank_high_bits_;

    AddressMappingScheme address_mapping_scheme_;
};

enum class KernelType { ADD, BN, RELU, GEMV, MUL, GEMVTREE };
#endif
