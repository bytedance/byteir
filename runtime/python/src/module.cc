//===- module.cc ----------------------------------------------*--- C++ -*-===//
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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "brt/core/common/common.h"
#include "brt/core/common/logging/sinks/cerr_sink.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"

#include "brt/core/context/work_queue.h"

#ifdef USE_CUDA
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace brt::cuda;
#endif

#include <memory>
#include <optional>

#ifndef MODULE_NAME
#define MODULE_NAME _brt
#endif

namespace py = pybind11;
using namespace brt;

// TODO: python bindings should be built with exceptions on, use cmake flag and
// corresponding macro control it instead of call throw directly
#define THROW_ON_FAIL(expr)                                                    \
  do {                                                                         \
    auto status = (expr);                                                      \
    if (!status.IsOK()) {                                                      \
      throw std::runtime_error(status.ToString());                             \
    }                                                                          \
  } while (0)

namespace {
class ReqeustContextWithSession {
public:
  auto Run() { return session->Run(Context()); }
  RequestContext &Context() {
    if (!req) {
      throw std::runtime_error("Unintialize request context");
    }
    return *req;
  }
  ReqeustContextWithSession(std::shared_ptr<Session> session_,
                            WorkQueue *work_queue)
      : session(session_) {
    THROW_ON_FAIL(session->NewRequestContext(&req, work_queue));
  }
  ~ReqeustContextWithSession() { req.reset(); }

private:
  std::unique_ptr<RequestContext> req;
  std::shared_ptr<Session> session;
};

class PyEnv {
public:
  static PyEnv *GetInstance();

  // TODO disable this for minimal build
  logging::LoggingManager *GetLoggingManager() const {
    return logging_manager_.get();
  }

private:
  PyEnv();

  std::string name_;

  std::unique_ptr<brt::logging::LoggingManager> logging_manager_;
};

PyEnv *PyEnv::GetInstance() {
  static PyEnv instance;
  return &instance;
}

PyEnv::PyEnv() {
  name_ = "PyEnvLoggingManager";

  logging_manager_ = std::make_unique<brt::logging::LoggingManager>(
      std::unique_ptr<brt::logging::ISink>{
          new brt::logging::CErrSink{}} /*sink*/,
      brt::logging::Severity::kWARNING, false,
      brt::logging::LoggingManager::InstanceType::Default, &name_);
}

class PyCustomAllocator final : public IAllocator {
public:
  PyCustomAllocator(py::object alloc_f, py::object free_f)
      : IAllocator(BrtMemoryInfo("PyCustom", "cuda", BrtCustomAllocator)),
        alloc_f_(alloc_f), free_f_(free_f) {}
  void *Alloc(size_t size) override {
    py::gil_scoped_acquire _;
    return reinterpret_cast<void *>(py::cast<size_t>(alloc_f_(size)));
  }
  void Free(void *p) override {
    if (Py_IsInitialized() == 0)
      return;

    py::gil_scoped_acquire _;
    free_f_(reinterpret_cast<size_t>(p));
  }

private:
  py::object alloc_f_;
  py::object free_f_;
};
} // namespace

PYBIND11_MODULE(MODULE_NAME, m) {
  // initialize internal logger
  static_cast<void>(PyEnv::GetInstance());

  py::class_<ReqeustContextWithSession,
             std::unique_ptr<ReqeustContextWithSession>>(m, "RequestContext")
      .def("bind_arg",
           [](ReqeustContextWithSession &req, size_t offset, const size_t ptr) {
             THROW_ON_FAIL(
                 req.Context().BindArg(offset, reinterpret_cast<void *>(ptr)));
           })
      .def("get_arg",
           [](ReqeustContextWithSession &req, size_t offset) {
             void *ptr = req.Context().GetArg(offset);
             return reinterpret_cast<size_t>(ptr);
           })
      .def(
          "finish_io_binding",
          [](ReqeustContextWithSession &req) {
            req.Context().FinishIOBinding();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sync",
          [](ReqeustContextWithSession &req) {
            THROW_ON_FAIL(req.Context().Sync());
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "run",
          [](ReqeustContextWithSession &req) { THROW_ON_FAIL(req.Run()); },
          py::call_guard<py::gil_scoped_release>());

  py::class_<Session, std::shared_ptr<Session>>(m, "Session")
      .def(py::init([](const std::string &device, py::object alloc_f,
                       py::object free_f) {
             auto session = std::make_shared<Session>();
             if (device != "CUDA") {
               throw std::runtime_error("unsupported device type " + device);
             }
#ifdef USE_CUDA
             else {
               if (!alloc_f || !free_f) {
                 THROW_ON_FAIL(
                     DefaultCUDAExecutionProviderFactory(session.get()));
               } else {
                 auto provider = std::make_unique<CUDAExecutionProvider>();
                 auto allocator =
                     std::make_unique<PyCustomAllocator>(alloc_f, free_f);
                 session->AddAllocator(std::move(allocator));
                 session->AddExecutionProvider(std::move(provider));
               }
             }
#endif
             return session;
           }),
           py::arg("device") = "CUDA", py::arg("alloc_func") = py::none(),
           py::arg("free_func") = py::none())
      .def(
          "load",
          [](Session &session, const std::string &path,
             const std::string &fmt) {
            THROW_ON_FAIL(session.Load(path, fmt));
          },
          py::arg("path"), py::arg("format") = "byre")
      .def(
          "new_request_context",
          [](std::shared_ptr<Session> session, std::optional<size_t> stream) {
            std::unique_ptr<WorkQueue> work_queue;
#ifdef USE_CUDA
            if (stream.has_value()) {
              work_queue.reset(new CUDAExternalStreamWorkQueue(
                  reinterpret_cast<CUstream_st *>(stream.value())));
            } else { // use cuda default stream
              int device_id;
              BRT_CUDA_CHECK(cudaGetDevice(&device_id));
              work_queue.reset(new CUDAWorkQueue(device_id));
            }
#endif
            return std::make_unique<ReqeustContextWithSession>(
                session, work_queue.release());
          },
          py::arg("stream") = py::none())
  // clang-format off
#define DEF_SESSION_METH_GENERIC(name, impl)                                   \
  .def(#name, &Session::impl)
      DEF_SESSION_METH_GENERIC(get_arg_num, GetArgNum)
      DEF_SESSION_METH_GENERIC(get_weight_num, GetWeightNum)
      DEF_SESSION_METH_GENERIC(get_weight_names, GetWeightNames)
      DEF_SESSION_METH_GENERIC(get_input_names, GetInputNames)
      DEF_SESSION_METH_GENERIC(get_output_names, GetInputNames)
      DEF_SESSION_METH_GENERIC(get_weight_arg_offsets, GetWeightArgOffsets)
      DEF_SESSION_METH_GENERIC(get_input_arg_offsets, GetInputArgOffsets)
      DEF_SESSION_METH_GENERIC(get_output_arg_offsets, GetOutputArgOffsets)
      DEF_SESSION_METH_GENERIC(get_static_shape, GetStaticShape)
      DEF_SESSION_METH_GENERIC(get_graph_arg_alias_offset, GetGraphArgAliasOffset)
#undef DEF_SESSION_METH_GENERIC
      // clang-format on
      ;
}
