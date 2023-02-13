//===- event.h ------------------------------------------------*--- C++ -*-===//
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

#include <functional>
#include <memory>
#include <vector>

namespace brt {
class OpKernelInfo;

struct Events {
  struct BeforeExecutionPlanRun {
    static constexpr uint8_t kIdx = 0;
  };

  struct AfterExecutionPlanRun {
    static constexpr uint8_t kIdx = 1;
  };

  struct BeforeOpKernelRun {
    static constexpr uint8_t kIdx = 2;
    const OpKernelInfo &info;
  };

  struct AfterOpKernelRun {
    static constexpr uint8_t kIdx = 3;
    const OpKernelInfo &info;
  };

  static constexpr uint8_t kSize = 4;

  template <typename T> using Listener = std::function<void(const T &)>;
};

class EventListenerManager {
public:
  template <typename T> void AddEventListener(Events::Listener<T> &&listener) {
    static_assert(T::kIdx < Events::kSize);

    auto sptr = std::make_shared<Events::Listener<T>>(std::move(listener));
    refkeepers.emplace_back(sptr);
    listeners[T::kIdx].emplace_back(sptr);
  }

  template <typename T> void SignalEvent(const T &event) {
    for (auto &&maybe : listeners[T::kIdx]) {
      if (auto listener = maybe.lock()) {
        reinterpret_cast<Events::Listener<T> *>(listener.get())
            ->
            operator()(event);
      }
    }
  }

private:
  std::vector<std::shared_ptr<void>> refkeepers;
  std::array<std::vector<std::weak_ptr<void>>, Events::kSize> listeners;
};

} // namespace brt
