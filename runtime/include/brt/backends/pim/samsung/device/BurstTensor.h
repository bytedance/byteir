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

#ifndef __BURSTTENSOR__HPP__
#define __BURSTTENSOR__HPP__

#include "Burst.h"
#include "FP16.h"
#include "tests/KernelAddrGen.h"
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace DRAMSim {

struct TensorBurstType {
  vector<unsigned long> shape;
  vector<float> data;
  vector<uint16_t> u16Data;
  vector<unsigned long> bShape;
  vector<BurstType> bData;
  enum precision { FP32, FP16 };

  BurstType &getBurst(int x, int y) { return bData[y * bShape[1] + x]; }
  BurstType &getBurst(int x) { return bData[x]; }

  void loadTobShape(double divisor) {
    for (int i = 0; i < shape.size(); i++) {
      if (i == shape.size() - 1)
        bShape.push_back(ceil(shape[i] / divisor));
      else
        bShape.push_back(shape[i]);
    }
  }

  void loadFp32(std::vector<float> &x) {
    data = std::vector<float>(x.size());

    loadTobShape((double)8);
    for (int i = 0; i < data.size(); i += 8) {
      BurstType burst(data[i], data[i + 1], data[i + 2], data[i + 3],
                      data[i + 4], data[i + 5], data[i + 6], data[i + 7]);
      bData.push_back(burst);
    }
  }

  void loadFp16(std::vector<float> &data) {
    u16Data = std::vector<uint16_t>(data.size());
    for (int i = 0; i < data.size(); i++) {
      u16Data[i] = convertF2H(data[i]);
    }
    // npy::LoadArrayFromNumpy(filename, shape, u16Data);
    loadTobShape((double)16);
    for (int i = 0; i < u16Data.size(); i += 16) {
      BurstType burst((u16Data[i]), (u16Data[i + 1]), (u16Data[i + 2]),
                      (u16Data[i + 3]), (u16Data[i + 4]), (u16Data[i + 5]),
                      (u16Data[i + 6]), (u16Data[i + 7]), (u16Data[i + 8]),
                      (u16Data[i + 9]), (u16Data[i + 10]), (u16Data[i + 11]),
                      (u16Data[i + 12]), (u16Data[i + 13]), (u16Data[i + 14]),
                      (u16Data[i + 15]));
      bData.push_back(burst);
    }
  }

  template <typename T> void loadFp16FromFp32(std::vector<float> &x) {
    // npy::LoadArrayFromNumpy(filename, shape, data);
    data = std::vector<float>(x.size());
    loadTobShape((double)16);
    for (int i = 0; i < data.size(); i += 16) {
      fp16 dataF2H[16];
      for (int j = 0; j < 16; j++) {
        dataF2H[j] = convertF2H(data[i + j]);
      }
      BurstType burst(dataF2H[0], dataF2H[1], dataF2H[2], dataF2H[3],
                      dataF2H[4], dataF2H[5], dataF2H[6], dataF2H[7],
                      dataF2H[8], dataF2H[9], dataF2H[10], dataF2H[11],
                      dataF2H[12], dataF2H[13], dataF2H[14], dataF2H[15]);
      bData.push_back(burst);
    }
  }

  void copyBurst(BurstType *b, unsigned long size) {
    bShape.push_back(size);

    for (int i = 0; i < size; i++) {
      BurstType burst;
      burst.set(b[i]);
      bData.push_back(burst);
    }
  }

  unsigned long getTotalDim() {
    unsigned long dim = 1;
    for (int i = 0; i < bShape.size(); i++) {
      dim *= bShape[i];
    }
    return dim;
  }
};

class TDataDim {
private:
  unsigned getPrecisionToByte() {
    switch (PIMConfiguration::getPIMPrecision()) {
    case INT8:
      return 1;
    case FP16:
      return 2;
    case FP32:
      return 4;
    default:
      return 0;
    }
  }

  void loadData(KernelType kn_type, vector<float> &input_row0,
                vector<float> &input_row1, vector<float> &result_row) {
    string input_dim_str = to_string(input_dim_);

    switch (kn_type) {
    case KernelType::GEMV:

      input_npbst_.loadFp16(input_row0);
      input1_npbst_.loadFp16(input_row1);
      output_npbst_.loadFp16(result_row);

      output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
      input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
      input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());
      return;

    case KernelType::ADD: {
      input_npbst_.loadFp16(input_row0);
      input1_npbst_.loadFp16(input_row1);
      output_npbst_.loadFp16(result_row);

      output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
      input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
      input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

      return;
    }
    case KernelType::MUL: {
      input_npbst_.loadFp16(input_row0);
      input1_npbst_.loadFp16(input_row1);
      output_npbst_.loadFp16(result_row);

      output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
      input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
      input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

      return;
    }
    // case KernelType::RELU: {
    //   input_npbst_.loadFp16("data/relu/relu_input_" + input_dim_str +
    //   ".npy"); output_npbst_.loadFp16("data/relu/relu_output_" +
    //   input_dim_str + ".npy");

    //   output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
    //   input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());

    //   return;
    // }
    default: {
      ERROR("== Error - Unknown KernelType trying to load data");
      exit(-1);
      return;
    }
    }
  }

  void loadDummyData(KernelType kn_type) {
    switch (kn_type) {
    case KernelType::GEMV:
    case KernelType::GEMVTREE: {
      weight_npbst_.shape.push_back(output_dim_);
      weight_npbst_.shape.push_back(input_dim_);
      weight_npbst_.loadTobShape(16);

      input_npbst_.shape.push_back(batch_size_);
      input_npbst_.shape.push_back(input_dim_);
      input_npbst_.loadTobShape(16);

      for (int i = 0; i < input_npbst_.bShape[1]; i++) {
        BurstType null_bst;
        null_bst.set((float)0);
        input_npbst_.bData.push_back(null_bst);
      }

      return;
    }
    case KernelType::ADD:
    case KernelType::MUL:
    case KernelType::RELU: {
      input_npbst_.shape.push_back(batch_size_);
      input_npbst_.shape.push_back(input_dim_);
      input_npbst_.loadTobShape(16);

      output_npbst_.shape.push_back(batch_size_);
      output_npbst_.shape.push_back(output_dim_);
      output_npbst_.loadTobShape(16);

      return;
    }
    default: {
      return;
    }
    }
  }

public:
  /* data */
  TensorBurstType input_npbst_;
  TensorBurstType input1_npbst_;
  TensorBurstType weight_npbst_;
  TensorBurstType output_npbst_;

  /* dump */
  TensorBurstType preloaded_npbst_;
  TensorBurstType result_npbst_;
  TensorBurstType reduced_result_npbst_;

  size_t burst_cnt_;
  TensorBurstType *preloaded_bst_;
  TensorBurstType *result_;
  TensorBurstType *reduced_result_;

  /* dim */
  unsigned long output_dim_;
  int input_dim_;
  int input1_dim_;
  int batch_size_;
  bool used_data_;

  TDataDim(KernelType kn_type, uint32_t batch_size, uint32_t output_dim,
           uint32_t input_dim, bool used_data, vector<float> &input_row0,
           vector<float> &input_row1, vector<float> &result_row) {
    batch_size_ = batch_size;
    output_dim_ = output_dim;
    input_dim_ = input_dim;
    used_data_ = used_data;

    switch (kn_type) {
    case KernelType::MUL:
    case KernelType::ADD: {
      input1_dim_ = input_dim;
      break;
    }
    default: {
      break;
    }
    }

    // load data from files
    if (used_data_)
      loadData(kn_type, input_row0, input_row1, result_row);
    else
      loadDummyData(kn_type);
  }

  uint32_t getDataSize(uint32_t dim1, uint32_t dim2 = 1, uint32_t dim3 = 1) {
    return dim1 * dim2 * dim3 * getPrecisionToByte();
  }

  void printDim(KernelType kn_type) {
    switch (kn_type) {
    case KernelType::GEMV:
    case KernelType::GEMVTREE: {
      cout << "  Weight data dimension : " << output_dim_ << "x" << input_dim_
           << endl;
      if (batch_size_ > 1) {
        cout << "  Input data dimension : " << input_dim_ << "x" << batch_size_
             << endl;
        cout << "  Output data dimension : " << output_dim_ << "x"
             << batch_size_ << endl;
      } else {
        cout << "  Input data dimension : " << input_dim_ << endl;
        cout << "  Output data dimension : " << output_dim_ << endl;
      }
      break;
    }
    case KernelType::MUL:
    case KernelType::ADD:
    case KernelType::RELU: {
      cout << "  Input/output data dimension : " << output_dim_ << endl;
      break;
    }
    default: {
      ERROR("== Error - Unknown KernelType trying to load data");
      exit(-1);
      break;
    }
    }
  }
  uint32_t getNumElementsPerBlocks() {
    return ((getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") *
             getConfigParam(UINT, "BL") / 8) /
            getPrecisionToByte());
  }

  uint32_t dimTobShape(int in_dim) {
    return ceil(in_dim / getNumElementsPerBlocks());
  }

  uint32_t bShape1ToDim(int bSahpe1) {
    return bSahpe1 * getNumElementsPerBlocks();
  }
};

} // namespace DRAMSim
#endif