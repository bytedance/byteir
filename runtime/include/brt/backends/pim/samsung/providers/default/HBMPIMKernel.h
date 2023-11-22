/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 *distributed, transmitted, transcribed, stored in a retrieval system, or
 *translated into any human or computer language in any form by any
 *means,electronic, mechanical, manual or otherwise, or disclosed to third
 *parties without the express written permission of Samsung Electronics. (Use of
 *the Software is restricted to non-commercial, personal or academic, research
 *purpose only)
 **************************************************************************************************/

#ifndef __PIM_KERNEL_HPP__
#define __PIM_KERNEL_HPP__

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "brt/backends/pim/samsung/device/BurstTensor.h"
#include "tests/KernelAddrGen.h"

using namespace std;
using namespace DRAMSim;

class HBMPIMKernel {
public:
  HBMPIMKernel(shared_ptr<MultiChannelMemorySystem> mem, int num_pim_chan,
               int num_pim_rank)
      : mem_(mem), num_pim_chans_(num_pim_chan), num_pim_ranks_(num_pim_rank),
        mode_(PIMConfiguration::getPIMMode()),
        num_banks_(getConfigParam(UINT, "NUM_BANKS")),
        num_pim_blocks_(getConfigParam(UINT, "NUM_PIM_BLOCKS")),
        num_bank_groups_(getConfigParam(UINT, "NUM_BANK_GROUPS")),
        use_all_grf_(false), srf_bst_(NULL), cycle_(0) {
    transaction_size_ =
        getConfigParam(UINT, "BL") *
        (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8); // in byte

    // FIXME: HARDCODED
    num_grf_ = num_grfA_ = num_grfB_ = 8;
    num_total_pim_blocks_ = num_pim_blocks_ * num_pim_chans_ * num_pim_ranks_;

    pim_chans_.clear();
    for (int i = 0; i < num_pim_chans_; i++)
      pim_chans_.push_back(i);

    pim_ranks_.clear();
    for (int i = 0; i < num_pim_ranks_; i++)
      pim_ranks_.push_back(i);

    pim_addr_mgr_ = make_shared<PIMAddrManager>(num_pim_chan, num_pim_rank);
  }

  int transaction_size_;
  int num_pim_chans_, num_pim_ranks_;
  int num_grfA_, num_grfB_, num_grf_;
  bool use_all_grf_;
  shared_ptr<PIMAddrManager> pim_addr_mgr_;

  void addBarrier();
  void runPIM();
  uint64_t getCycle();
  void parkIn();
  void parkOut();
  void changePIMMode(dramMode mode1, dramMode mode2);
  void addTransactionAll(bool isWrite, int bg, int bank, int row, int col,
                         const std::string tag, BurstType *bst,
                         bool use_barrier = false, int num_loop = 1);
  void addTransactionAll(bool isWrite, int bg, int bank, int row, int col,
                         BurstType *bst, bool use_barrier = false,
                         int num_loop = 1);
  /*
  void preprocessBn(TensorBurstType* mean_npbst, TensorBurstType* var_npbst,
                    TensorBurstType* gamma_npbst, TensorBurstType* beta_npbst,
                    TensorBurstType* input_npbst, fp16** params, float eps);
  void preprocessSrf(TensorBurstType* input_npbst, fp16** params, int
  burst_offset, int num_srf_usage);
  */
  /*
  void programSrf();
  */
  void programCrf(vector<PIMCmd> &cmds);
  void setCrf(BurstType *bst, bool op, bool use_all_grf, int ctc,
              bool grfA_zero, bool grfB_zero);
  unsigned getResultColGemv(int input_dim, int output_dim);
  void changeBank(pimBankType bank_types, int &cidx, int &rank, int &bg,
                  int &bank, unsigned &startingRow, unsigned &startingCol,
                  unsigned &row, unsigned &col);
  void preloadGemv(TensorBurstType *operand, unsigned starting_row = 0,
                   unsigned starting_col = 0);
  void preloadNoReplacement(TensorBurstType *operand, unsigned startingRow,
                            unsigned startingCol);
  /*
  void preloadEltwise(TensorBurstType* operand, pimBankType bank_types, unsigned
  startingRow, unsigned startingCol);
  */
  void executeGemv(TensorBurstType *w_data, TensorBurstType *i_data,
                   bool is_tree);
  void executeEltwise(int dim, pimBankType bank_types, KernelType ktype,
                      int input0_row, int result_row, int input1_row = 0);
  void computeGemv(TensorBurstType *data, int num_input_tiles,
                   int num_output_tile, int input_tile, int output_tile,
                   int batch_idx, pimBankType bank_types);
  void computeAddOrMul(int numTile, int input0Row, int resultRow,
                       int input1Row);
  void computeRelu(int numTile, int input0Row, int resultRow);
  // void computeBn(int numTile, int input0Row, int resultRow);

  void readResult(BurstType *resultBst, pimBankType bank_types, int output_dim,
                  uint64_t baseAddr = 0, unsigned startingRow = 0,
                  unsigned startingCol = 0);
  void readData(BurstType *bst_data, size_t bst_cnt, unsigned s_row = 0,
                unsigned s_col = 0);
  void adderTree(BurstType *result, int output_dim, int numTile, int step,
                 fp16 *temp);

private:
  unsigned cycle_;
  unsigned num_banks_, num_pim_blocks_, num_bank_groups_, num_total_pim_blocks_;
  BurstType null_bst_, bst_hab_pim_, bst_hab_;
  BurstType crf_bst_[4];
  BurstType *srf_bst_;
  vector<int> pim_chans_;
  vector<int> pim_ranks_;
  PIMMode mode_;
  shared_ptr<MultiChannelMemorySystem> mem_;
};

#endif