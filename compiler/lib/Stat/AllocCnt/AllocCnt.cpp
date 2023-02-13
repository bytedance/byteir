//===- AllocCnt.cpp -------------------------------------------------------===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Stat/AllocCnt/AllocCnt.h"

#include "byteir/Analysis/UseRange.h"
#include "byteir/Stat/Common/Reg.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

namespace {

bool isAllocation(Operation *op) { return llvm::isa<memref::AllocOp>(op); }

bool isStaticAllocation(Value value) {
  if (MemRefType t = value.getType().dyn_cast_or_null<MemRefType>()) {
    return isStatic(t);
  }
  return false;
}

size_t getSizeInBytes(Value value) {
  MemRefType t = value.getType().cast<MemRefType>();
  auto sizeInBits = getSizeInBits(t);
  assert(sizeInBits.has_value());
  return (*sizeInBits + 7) >> 3;
}

struct Event {
  size_t timestamp;
  int op;
  size_t length;
  bool operator<(const Event &other) const {
    if (this->timestamp != other.timestamp) {
      return this->timestamp < other.timestamp;
    }
    return this->op < other.op;
  }
};

std::string allocCntStatisticsSingle(func::FuncOp func) {
  size_t staticAlloc = 0, dynamicAlloc = 0;
  size_t totalMemory = 0, peakMemory = 0;
  int currentMemory = 0;
  std::vector<Event> events;

  byteir::Liveness liveness(func);
  mlir::bufferization::BufferPlacementAllocs allocs(func);
  mlir::BufferViewFlowAnalysis aliases(func);
  byteir::UserangeAnalysis useRange(func, &liveness, allocs, aliases);

  func.walk([&](Operation *op) {
    if (isAllocation(op)) {
      auto value = op->getResult(0);
      if (isStaticAllocation(value)) {
        ++staticAlloc;
        size_t length = getSizeInBytes(value);
        totalMemory += length;
        if (auto uses = useRange.getUserangeInterval(value)) {
          for (auto &&it : **uses) {
            // when use start,increase 1
            // and when use end + 1, decrease 1
            events.push_back({it.start, 1, length});
            events.push_back({it.end + 1, -1, length});
          }
        }
      } else {
        ++dynamicAlloc;
      }
    }
  });

  std::sort(events.begin(), events.end());
  for (auto &&e : events) {
    currentMemory += e.op * e.length;
    assert(currentMemory >= 0);
    peakMemory = std::max(peakMemory, static_cast<size_t>(currentMemory));
  }
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os << "-======= For function " << func.getName() << " =======-\n";
  os << "  nr_static_allocation = " << staticAlloc << "\n"
     << "  nr_dynamic_allocation = " << dynamicAlloc << "\n"
     << "  total_static_allocated_memory = " << totalMemory << "\n"
     << "  peak_static_memory = " << peakMemory << "\n";
  return ret;
}
} // namespace

//===----------------------------------------------------------------------===//
// AllocCnt registration
//===----------------------------------------------------------------------===//

void byteir::registerAllocCntStatistics() {
  MLIRStatRegistration reg("alloc-cnt",
                           [](ModuleOp module, raw_ostream &output) {
                             return byteir::allocCntStatistics(module, output);
                           });
}

mlir::LogicalResult byteir::allocCntStatistics(ModuleOp moduleOp,
                                               llvm::raw_ostream &os) {
  for (func::FuncOp func : moduleOp.getOps<func::FuncOp>()) {
    os << allocCntStatisticsSingle(func) << "\n";
  }

  return success();
}