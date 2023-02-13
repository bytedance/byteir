//===- execution_frame.h --------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/framework/value.h"
#include "brt/core/ir/graph_info.h"
#include "brt/core/ir/ir.h"
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brt {

// Forward decl
class IAllocator;

struct GroupAllocationHook {
  // indicate which tensors should be allocated and freed, the length of
  // tensor_indexes should be the same with the length of the return value of
  // the alloc_f and the length of the argument of the free_f
  std::vector<size_t> tensor_indexes;
  // return an array of AsyncValues, each AsyncValue is mapping to the
  // corresponding tensor index
  // Note: this method could be called more than once due to mulitple execution
  // frameworks
  std::function<std::vector<AsyncValue>(void)> alloc_f;
  // receive an array of AsyncValues as argument, each AsyncValue is mapping to
  // the corresponding tensor index
  // Note: this method could be called more than once due to multiple execution
  // frameworks
  std::function<void(std::vector<AsyncValue>)> free_f;
};

// TODO: move to common header
class Scalar {
public:
  Scalar() = default;

  template <typename T> Scalar(const T &data) { Set(data); }

  template <typename T> T Get() const {
    BRT_ENFORCE(dtype_enum_v<T> == dtype);
    return reinterpret_cast<const T &>(data);
  }

  template <typename T> void Set(const T &newData) {
    static_assert(std::is_standard_layout_v<T> && std::is_trivial_v<T>);
    dtype = dtype_enum_v<T>;
    reinterpret_cast<T &>(data) = newData;
  }

private:
  DTypeEnum dtype;
  std::aligned_storage_t<sizeof(size_t), alignof(size_t)> data;
};

/**
 * ExecutionFrame is a abstract class that holds all tensors,
 * including inputs, outputs, and constant variabls, like initializers.
 *
 * It can be derived into a class that separate buffers
 * between stateful variables and initializeres.
 *
 * It can be also derived into a class using static memory planning.
 */

/**
 * The base class of ExecutionFrame
 */
class ExecutionFrame {
public:
  // StateInfo holds state for an ExecutionFrame
  // E.g. cublas handle offset or temp buffer
  struct StateInfo {

  private:
    std::shared_mutex mutex;
    std::unordered_map<std::string, size_t> name_to_offset;

    size_t GetStateOffsetUnsafe(const std::string &key) {
      auto &&it = name_to_offset.find(key);
      return it->second;
    }

  public:
    template <typename FUNC>
    common::Status CreateStateIfNotExist(const std::string &key,
                                         ExecutionFrame *frame, FUNC functor) {
      // test and test-and-set
      // 1st test
      std::shared_lock<std::shared_mutex> lock_read(mutex);
      auto found_1 = name_to_offset.find(key);
      if (found_1 == name_to_offset.end()) {
        // if offset not exit
        lock_read.unlock();

        // Note every frame without state,
        // needs to create a state.
        // even two concurrent frames can create their own private state.
        auto state = functor();
        size_t offset = frame->PushBackState(state);

        // lock the region, and allow only the first frame to enter
        // Note this lock is used to protect offset
        std::unique_lock<std::shared_mutex> lock_write(mutex);
        // 2nd test in a lock
        auto found_2 = name_to_offset.find(key);
        if (found_2 == name_to_offset.end()) {
          // set value
          name_to_offset.emplace(key, offset);
        }
      } else if (found_1->second >= frame->StateSize()) {
        lock_read.unlock();
        // if offset exist, but state not exist
        // meaning the current frame is a new frame,
        // but the first frame in a session.
        // so just creat a handle here
        auto state = functor();
        frame->PushBackState(state);
      }

      return Status::OK();
    }

    size_t GetStateOffset(const std::string &key) {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return GetStateOffsetUnsafe(key);
    }

    bool HasState(const std::string &key) {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return name_to_offset.find(key) != name_to_offset.end();
    }
  };

  struct InternalState {
    /*
     * BeforePrologue
     *  |     |
     *  |   (call prologue)
     *  |     |
     *  |     +----------+
     *  |     |      (run model)
     *  |     V          |
     *  |   MainLoop-----+
     *  |     |
     *  |   (call epilogue)
     *  |     |
     *  V     V
     * AfterEpilogue
     */
    static constexpr uint8_t BeforePrologue = 0;
    static constexpr uint8_t MainLoop = 1;
    static constexpr uint8_t AfterEpilogue = 2;
    static constexpr uint8_t Error = 31;
    static constexpr uint8_t Unknown = 255;
    struct SingleStateTransition {
      // early bind given ExecutionFrame on this transition so we only need
      // save the edge from current frame internal state
      SingleStateTransition(ExecutionFrame &frame)
          : frame_(frame), from_(frame_.internal_state_.cur_state),
            to_(Unknown) {}
      SingleStateTransition &&
      Edge(uint8_t from, uint8_t to,
           std::function<common::Status(void)> &&cb) && {
        if (Unknown == to_ && from == from_) {
          to_ = to;
          cb_ = cb;
        }
        return std::move(*this);
      }
      SingleStateTransition &&Edge(uint8_t from, uint8_t to) && {
        static auto cb = [] { return common::Status::OK(); };
        return std::move(*this).Edge(from, to, cb);
      }
      SingleStateTransition &&Invariant(uint8_t s) && {
        return std::move(*this).Edge(s, s);
      }
      common::Status Apply() && {
        uint8_t &cur_state = frame_.internal_state_.cur_state;
        BRT_ENFORCE(cur_state == from_, "internal state of execution frame "
                                        "was changed during state tansition");
        if (to_ != Unknown) {
          auto err = cb_();
          if (err.IsOK()) {
            cur_state = to_;
          } else {
            cur_state = Error;
          }
          return err;
        }
        cur_state = Error;
        return common::Status(common::StatusCategory::BRT,
                              common::StatusCode::FAIL);
      }
      ExecutionFrame &frame_;
      const uint8_t from_;
      uint8_t to_;
      std::function<common::Status(void)> cb_;
    };
    uint8_t cur_state = 0;
  };

  ExecutionFrame() {}

  virtual ~ExecutionFrame();

  // TODO confirm API with services
#if 0
  virtual common::Status Init(const std::vector<int>& feed_mlvalue_idxs, 
            const std::vector<AsyncValueRef>& feeds,
            const std::unordered_map<int, AsyncValue>& initializers,
            const std::vector<AsyncValue>& fetches) = 0;
#endif

  virtual AsyncValueRef GetAsyncValueRef(size_t) const = 0;
  virtual AsyncValue GetAsyncValue(size_t) const = 0;
  virtual ShapeRef GetShapeRef(size_t) const = 0;
  virtual Shape GetShape(size_t) const = 0;
  virtual common::Status SetShape(size_t, const Shape &) = 0;

  virtual void FinishIOBinding() = 0;
  virtual void AllocIntermediate() = 0;
  virtual void BindArg(size_t idx, const void *) = 0;
  virtual void *GetArg(size_t) = 0;

  // TODO: unify tensor and scalar to generic Value class
  template <typename T> T GetScalar(size_t idx) {
    return GetScalarImpl(idx).Get<T>();
  }
  template <typename T> common::Status SetScalar(size_t idx, const T &data) {
    Scalar scalar(data);
    return SetScalarImpl(idx, scalar);
  }
  virtual Scalar GetScalarImpl(size_t) = 0;
  virtual common::Status SetScalarImpl(size_t, const Scalar &) = 0;

  auto GetIStateTransition() {
    return InternalState::SingleStateTransition(*this);
  }

  void *GetState(size_t id) {
    if (id < meta_states_.size()) {
      return meta_states_[id];
    }
    return nullptr;
  }

  void *GetAndResetState(size_t id) {
    if (id < meta_states_.size()) {
      void *ptr = meta_states_[id];
      meta_states_[id] = nullptr;
      return ptr;
    }
    return nullptr;
  }

  size_t PushBackState(void *ptr) {
    size_t id = meta_states_.size();
    meta_states_.push_back(ptr);
    return id;
  }

  size_t StateSize() { return meta_states_.size(); }

protected:
  InternalState internal_state_;
  std::vector<void *> meta_states_;

private:
  ExecutionFrame(const ExecutionFrame &) = delete;
  ExecutionFrame &operator=(const ExecutionFrame &) = delete;
  ExecutionFrame(ExecutionFrame &&) = delete;
  ExecutionFrame &operator=(ExecutionFrame &&) = delete;
};

/**
 * The default BRT inference ExecutionFrame that shares initializers and
 * privatizes stateful variables.
 *
 * The layout of BRTInferenceExecutionFrame is assigned as the following
 * sequence
 *
 * 1. Graph Weights (stored in Info, Context)
 * 2. Graph Inputs  (stored in Context)
 * 3. Graph Outputs (stored in Context)
 * 4. Intermediate Tensors  (stored Context)
 *
 * BRT Inference ExecutionFrame allows weight override.
 * Therefore, info will only copy non-override weights to Context.
 */

class BRTInferenceExecutionFrame : public ExecutionFrame {
public:
  /**
   * ConstructInfo is a structure holding information to create an
   * ExecutionFrame
   */
  // TODO change to multiple allocator support
  // TODO promote this to outer namespace
  struct ConstructInfo {
    // uninitalized memory offset
    static constexpr uint64_t kUninitializedMemOffset =
        std::numeric_limits<uint64_t>::max();
    // memory offset which indicates the shape of value is dynamic and can't
    // allocate memory for it statically
    static constexpr uint64_t kDynamicMemOffset =
        std::numeric_limits<uint64_t>::max() - 1;

    // uninitialized allocator offset
    static constexpr int64_t kUninitializedAllocatorOffset = -2;
    // allocator offset which indicates the underlying memory of corresponding
    // intermediate tensor is allocated by group allocation
    static constexpr int64_t kGroupAllocationOffset = -1;

    const brt::ir::GraphInfo &graph_info;

    std::vector<IAllocator *> allocators;

    std::unordered_map<std::string, size_t> space_to_allocator_id;

    // sum all intermeidates in bytes
    std::vector<uint64_t> total_intermediate_sizes;
    // uint64_t total_intermediate_size = 0;

    // store all offsets of all intermeidates,
    // which is used for allocating entire buffers
    // a pair {allocators offset, the offset in that base}
    // e.g. {0, 1024} => the 0th allocator's base + 1024
    std::vector<std::pair<int64_t, uint64_t>> intermediate_ids_and_offsets;

    // store all buffers of weights
    std::vector<AsyncValue> weights;

    std::vector<IAllocator *> weight_and_ios_allocators;

    // each dynamic allocation request is a pair of (intermediate_index,
    // dimensions), each dimension is an integer, which is the dimension size
    // for stataic dimension or represents the corresponding runtime scalar
    // index for dynamic size dimension
    std::vector<std::pair<size_t, std::vector<int64_t>>>
        dynamic_allocation_requests;

    std::vector<std::unique_ptr<GroupAllocationHook>> group_allocation_hooks;

    ConstructInfo(const brt::ir::GraphInfo &info) : graph_info(info) {}
  };

  BRTInferenceExecutionFrame(const ConstructInfo &info);

  virtual ~BRTInferenceExecutionFrame();

  AsyncValueRef GetAsyncValueRef(size_t) const override;
  AsyncValue GetAsyncValue(size_t) const override;
  ShapeRef GetShapeRef(size_t) const override;
  Shape GetShape(size_t) const override;
  common::Status SetShape(size_t, const Shape &) override;

  void FinishIOBinding() override;
  void AllocIntermediate() override;
  void BindArg(size_t idx, const void *) override;
  void *GetArg(size_t) override;

  Scalar GetScalarImpl(size_t) override;
  common::Status SetScalarImpl(size_t, const Scalar &) override;

  uint64_t GetBytes(size_t);

private:
  // TODO change to multiple allocator support
  struct FrameContext {
    std::vector<void *> intermediate_base_addresses;
    std::vector<void *> weights_and_ios; // weight, inputs, and outputs
    std::vector<void *> intermediate_values;
    std::vector<bool> is_io_allocated;
    std::vector<std::optional<Shape>> static_shapes;
    std::vector<Scalar> scalars;
  };

  const ConstructInfo &info_;

  FrameContext ctx_;

  BRTInferenceExecutionFrame(const BRTInferenceExecutionFrame &) = delete;
  BRTInferenceExecutionFrame &
  operator=(const BRTInferenceExecutionFrame &) = delete;
  BRTInferenceExecutionFrame(BRTInferenceExecutionFrame &&) = delete;
  BRTInferenceExecutionFrame &operator=(BRTInferenceExecutionFrame &&) = delete;
};

} // namespace brt
