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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "brt/core/common/common.h"
#include "brt/core/common/logging/sinks/cerr_sink.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"

#include "brt/core/context/work_queue.h"

#include "brt/backends/cpu/device/cpu_device_api.h"
#include "brt/backends/cpu/device/cpu_work_queue.h"
#include "brt/backends/cpu/providers/default/cpu_provider.h"

#ifdef BRT_USE_CUDA
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_device_api.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"

#ifdef BRT_USE_NCCL
#include "brt/backends/nccl/providers/nccl_provider.h"
#include "brt/core/distributed/rendezvous_socket.h"
#endif // BRT_USE_NCCL

#include <cuda.h>
#include <cuda_runtime.h>

using namespace brt::cuda;
#endif // BRT_USE_CUDA

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

// clang-format off
#define FOR_EACH_BRT_DTYPE(cb)                                                 \
  cb(DTypeEnum::Float32)                                                       \
  cb(DTypeEnum::Int32)                                                         \
  cb(DTypeEnum::Int64)                                                         \
  cb(DTypeEnum::UInt8)                                                         \
  cb(DTypeEnum::UInt32)                                                        \
  cb(DTypeEnum::Float16)                                                       \
  cb(DTypeEnum::Float64)                                                       \
  cb(DTypeEnum::Bool)                                                          \
  cb(DTypeEnum::Int8)                                                          \
  cb(DTypeEnum::Int16)                                                         \
  cb(DTypeEnum::UInt16)                                                        \
  cb(DTypeEnum::UInt64)
// clang-format on

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

#if defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)
class RequestContextWithDistributedSession {
public:
  auto Run() { return session->Run(Context()); }

  RequestContextWithDistributedSession(
      std::shared_ptr<DistributedSession> session_, WorkQueue *work_queue)
      : session(session_) {
    THROW_ON_FAIL(session->NewRequestContext(&req, work_queue));
  }

  RequestContextWithDistributedSession(
      std::shared_ptr<DistributedSession> session_)
      : session(session_) {
    THROW_ON_FAIL(session->NewRequestContext(&req));
  }

  RequestContext &Context() {
    if (!req) {
      throw std::runtime_error("Unintialize request context");
    }
    return *req;
  }

  ~RequestContextWithDistributedSession() { req.reset(); }

private:
  std::unique_ptr<RequestContext> req;
  std::shared_ptr<DistributedSession> session;
};
#endif // defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)

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
      : IAllocator(BrtMemoryInfo("PyCustom", "cuda",
                                 brt::BrtAllocatorType::CustomAllocator)),
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

using PyDType = DTypeEnum;

PyDType npdtype_to_pydtype(py::dtype dtype) {
#define Case(T)                                                                \
  if (dtype.is(py::dtype::of<DTypeTraits<T>::type_t>())) {                     \
    return T;                                                                  \
  }
  FOR_EACH_BRT_DTYPE(Case)
#undef Case
  throw std::runtime_error("unsupporetd data type");
}

py::dtype pydtype_to_npdtype(PyDType dtype) {
  switch (dtype) {
#define Case(T)                                                                \
  case T:                                                                      \
    return py::dtype::of<DTypeTraits<T>::type_t>();
    FOR_EACH_BRT_DTYPE(Case)
#undef Case
  default:
    throw std::runtime_error("unsupporetd data type");
  }
}

PYBIND11_MODULE(MODULE_NAME, m) {
  // initialize internal logger
  static_cast<void>(PyEnv::GetInstance());

  m.def("get_free_port", []() { return brt::GetFreePort(); });

  m.def("create_server", [](int nranks, int port) {
    THROW_ON_FAIL(brt::CreateServer(nranks, port));
    return;
  });

  py::enum_<PyDType>(m, "DType")
      .value("float32", PyDType::Float32)
      .value("int32", PyDType::Int32)
      .value("int64", PyDType::Int64)
      .value("uint8", PyDType::UInt8)
      .value("uint32", PyDType::UInt32)
      .value("float16", PyDType::Float16)
      .value("float64", PyDType::Float64)
      .value("bool", PyDType::Bool)
      .value("int8", PyDType::Int8)
      .value("int16", PyDType::Int16)
      .value("uint16", PyDType::UInt16)
      .value("uint64", PyDType::UInt64)
      .def(py::init(&npdtype_to_pydtype), py::arg("dtype"))
      .def("numpy", &pydtype_to_npdtype)
      .def("get_dtype",
           [](Session &session, size_t idx) { return session.GetDType(idx); });

  py::class_<ReqeustContextWithSession,
             std::unique_ptr<ReqeustContextWithSession>>(m, "RequestContext")
      .def("bind_arg",
           [](ReqeustContextWithSession &req, size_t offset, const size_t ptr) {
             THROW_ON_FAIL(
                 req.Context().BindArg(offset, reinterpret_cast<void *>(ptr)));
           })
      .def("bind_args",
           [](ReqeustContextWithSession &req, py::list offset_and_args) {
             for (auto handle : offset_and_args) {
               PyObject *obj = handle.ptr();
               if (!PyTuple_Check(obj) || PyTuple_Size(obj) != 2) {
                 PyErr_SetString(PyExc_TypeError,
                                 "expect pair of offset and arg");
                 return;
               }

               PyObject *offset = PyTuple_GetItem(obj, 0);
               PyObject *arg = PyTuple_GetItem(obj, 1);
               if (!PyLong_Check(offset)) {
                 PyErr_SetString(PyExc_TypeError, "offset should be integer");
                 return;
               }
               if (!PyLong_Check(arg)) {
                 PyErr_SetString(PyExc_TypeError, "arg should be integer");
                 return;
               }
               THROW_ON_FAIL(req.Context().BindArg(PyLong_AsSize_t(offset),
                                                   PyLong_AsVoidPtr(arg)));
             }
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

#if defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)
  py::class_<RequestContextWithDistributedSession,
             std::unique_ptr<RequestContextWithDistributedSession>>(
      m, "DistributedRequestContext")
      .def("bind_arg",
           [](RequestContextWithDistributedSession &req, size_t offset,
              const size_t ptr) {
             THROW_ON_FAIL(
                 req.Context().BindArg(offset, reinterpret_cast<void *>(ptr)));
           })
      .def("get_arg",
           [](RequestContextWithDistributedSession &req, size_t offset) {
             void *ptr = req.Context().GetArg(offset);
             return reinterpret_cast<size_t>(ptr);
           })
      .def(
          "finish_io_binding",
          [](RequestContextWithDistributedSession &req) {
            req.Context().FinishIOBinding();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sync",
          [](RequestContextWithDistributedSession &req) {
            THROW_ON_FAIL(req.Context().Sync());
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "run",
          [](RequestContextWithDistributedSession &req) {
            THROW_ON_FAIL(req.Run());
          },
          py::call_guard<py::gil_scoped_release>());
#endif // defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)

  py::class_<Session, std::shared_ptr<Session>>(m, "Session")
      .def(py::init([](const std::string &device, py::object alloc_f,
                       py::object free_f) {
             auto session = std::make_shared<Session>();
             if (device == "CPU") {
               // TODO Support customize cpu provider options.
               THROW_ON_FAIL(NaiveCPUExecutionProviderFactory(session.get()));
               THROW_ON_FAIL(CPUAllocatorFactory(session.get()));
               session->SetExecDevice(DeviceType::CPU, /*device_id*/ 0);
               session->AddDeviceAPI(DeviceType::CPU, GetCPUDeviceAPI());
             }
#ifdef BRT_USE_CUDA
             else if (device == "CUDA") {
               if (!alloc_f || !free_f) {
                 THROW_ON_FAIL(
                     DefaultCUDAExecutionProviderFactory(session.get()));
               } else {
                 auto provider = std::make_unique<CUDAExecutionProvider>();
                 auto allocator =
                     std::make_unique<PyCustomAllocator>(alloc_f, free_f);
                 int device_id;
                 BRT_CUDA_CHECK(cudaGetDevice(&device_id));
                 session->SetExecDevice(DeviceType::CUDA, device_id);
                 session->AddDeviceAPI(DeviceType::CUDA, GetCUDADeviceAPI());
                 session->AddAllocator(std::move(allocator));
                 session->AddExecutionProvider(std::move(provider));
               }
             }
#endif // BRT_USE_CUDA
             else {
               throw std::runtime_error("unsupported device type " + device);
             }

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
            if (session->GetDeviceType() == DeviceType::CPU) {
              work_queue.reset(new cpu::CPUNaiveWorkQueue());
            }
#ifdef BRT_USE_CUDA
            else if (session->GetDeviceType() == DeviceType::CUDA) {
              if (stream.has_value()) {
                work_queue.reset(new CUDAExternalStreamWorkQueue(
                    reinterpret_cast<CUstream_st *>(stream.value())));
              } else { // use cuda default stream
                int device_id;
                BRT_CUDA_CHECK(cudaGetDevice(&device_id));
                work_queue.reset(new CUDAWorkQueue(device_id));
              }
            }
#endif // BRT_USE_CUDA
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
      DEF_SESSION_METH_GENERIC(get_data_type, GetDType)
      DEF_SESSION_METH_GENERIC(get_graph_arg_alias_offset, GetGraphArgAliasOffset)
#undef DEF_SESSION_METH_GENERIC
      // clang-format on
      ;
#if defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)
  py::class_<DistributedSession, Session, std::shared_ptr<DistributedSession>>(
      m, "DistributedSession")
      .def(py::init([](int rank, int nrank, const std::string &host, int port) {
        auto distributed_session =
            std::make_shared<DistributedSession>(rank, nrank, host, port);
        THROW_ON_FAIL(CUDAAllocatorFactory(distributed_session.get(), rank));
        THROW_ON_FAIL(DefaultNCCLExecutionProviderFactory(
            distributed_session.get(), rank));
        return distributed_session;
      }))
      .def(
          "load_config",
          [](DistributedSession &session, std::vector<std::string> config) {
            std::string ir_url;
            session.LoadConfig(config, ir_url);
            return ir_url;
          },
          py::arg("config"))
      .def("new_request_context",
           [](std::shared_ptr<DistributedSession> session,
              std::optional<size_t> stream) {
             return std::make_unique<RequestContextWithDistributedSession>(
                 session);
           })
  // clang-format off
#define DEF_DISTRIBUTED_SESSION_METH_GENERIC(name, impl)                                   \
  .def(#name, &DistributedSession::impl)
      DEF_DISTRIBUTED_SESSION_METH_GENERIC(get_rank, GetRank)
      DEF_DISTRIBUTED_SESSION_METH_GENERIC(get_nranks, GetNRanks)
      DEF_DISTRIBUTED_SESSION_METH_GENERIC(get_host, GetHost)
      DEF_DISTRIBUTED_SESSION_METH_GENERIC(get_port, GetPort)
#undef DEF_DISTRIBUTED_SESSION_METH_GENERIC
      // clang-format on
      ;
#endif // defined(BRT_USE_CUDA) && defined(BRT_USE_NCCL)
}
