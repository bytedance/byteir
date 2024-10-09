//===- ByteIRModules.cpp ------------------------------------*--- C++ -*-===//
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

#include "bindings/c/Passes.h"
#include "byteir-c/Dialects.h"
#include "byteir-c/PDLValue.h"
#include "byteir-c/Passes.h"
#include "byteir-c/Translation.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

template <class Func, typename... Args>
py::object classmethod(Func f, Args... args) {
  py::object cf = py::cpp_function(f, args...);
  return py::reinterpret_borrow<py::object>((PyClassMethod_New(cf.ptr())));
}

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

static py::object wrap(MlirPDLValue pdlValue) {
  switch (pdlValue.kind) {
  case MlirPDLValueAttribute:
    return py::cast(mlirPDLValueCastToMlirAttribute(pdlValue));
  case MlirPDLValueOperation:
    return py::cast(mlirPDLValueCastToMlirOperation(pdlValue));
  case MlirPDLValueType:
    return py::cast(mlirPDLValueCastToMlirType(pdlValue));
  case MlirPDLValueTypeRange: {
    MlirType *types;
    intptr_t ntypes;
    mlirPDLValueCastToMlirTypeRange(pdlValue, &types, &ntypes);
    std::vector<MlirType> typeRange(types, types + ntypes);
    delete[] types;
    return py::cast(typeRange);
  }
  case MlirPDLValueValue:
    return py::cast(mlirPDLValueCastToMlirValue(pdlValue));
  case MlirPDLValueValueRange: {
    MlirValue *values;
    intptr_t nvalues;
    mlirPDLValueCastToMlirValueRange(pdlValue, &values, &nvalues);
    std::vector<MlirValue> valueRange(values, values + nvalues);
    delete[] values;
    return py::cast(valueRange);
  }
  }
  throw std::runtime_error("unknown pdl value kind");
}

static void emplaceResult(MlirPDLResultListRef pdlResults, py::object obj,
                          MlirPDLValueKind kind) {
  switch (kind) {
  case MlirPDLValueAttribute: {
    mlirPDLResultListEmplaceAttribute(pdlResults, obj.cast<MlirAttribute>());
    break;
  }
  case MlirPDLValueOperation: {
    mlirPDLResultListEmplaceOperation(pdlResults, obj.cast<MlirOperation>());
    break;
  }
  case MlirPDLValueType: {
    mlirPDLResultListEmplaceType(pdlResults, obj.cast<MlirType>());
    break;
  }
  case MlirPDLValueTypeRange: {
    std::vector<MlirType> types;
    for (auto &&elem : obj.cast<py::sequence>()) {
      types.push_back(elem.cast<MlirType>());
    }
    mlirPDLResultListEmplaceTypes(pdlResults, types.data(), types.size());
    break;
  }
  case MlirPDLValueValue: {
    mlirPDLResultListEmplaceValue(pdlResults, obj.cast<MlirValue>());
    break;
  }
  case MlirPDLValueValueRange: {
    std::vector<MlirValue> types;
    for (auto &&elem : obj.cast<py::sequence>()) {
      if (elem.ptr() != Py_None) {
        types.push_back(elem.cast<MlirValue>());
      } else {
        types.push_back(MlirValue{nullptr});
      }
    }
    mlirPDLResultListEmplaceValues(pdlResults, types.data(), types.size());
    break;
  }
  default:
    throw std::runtime_error("unknown pdl value kind");
  }
}

struct PyCtxGuard {
  PyCtxGuard(py::object obj) : ctx(std::move(obj)) {
    if (ctx)
      ctx.attr("__enter__")();
  }

  ~PyCtxGuard() {
    if (ctx)
      ctx.attr("__exit__")(py::none(), py::none(), py::none());
  }

  py::object ctx;
};

struct PyOpViewBuilderGuard {
  PyOpViewBuilderGuard(std::function<void(MlirOperation)> opInsertionCallback)
      : opViewCls(py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                      .attr("OpView")) {

    oldBuilder = opViewCls.attr("build_generic").attr("__func__");
    opViewCls.attr("build_generic") =
        classmethod([=](py::args args, py::kwargs kwargs) {
          py::object newOp = oldBuilder(*args, **kwargs);
          opInsertionCallback(py::cast<MlirOperation>(newOp.attr("operation")));
          return newOp;
        });
  }

  ~PyOpViewBuilderGuard() {
    opViewCls.attr("build_generic") =
        py::reinterpret_borrow<py::object>(PyClassMethod_New(oldBuilder.ptr()));
  }

  py::object opViewCls;
  py::object oldBuilder;
};

struct PyOperationBuilderGuard {
  PyOperationBuilderGuard(
      std::function<void(MlirOperation)> opInsertionCallback)
      : opCls(py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("Operation")) {
    oldBuilder = opCls.attr("create");
    opCls.attr("create") =
        py::cpp_function([=](py::args args, py::kwargs kwargs) {
          py::object newOp = oldBuilder(*args, **kwargs);
          opInsertionCallback(py::cast<MlirOperation>(newOp.attr("operation")));
          return newOp;
        });
  }

  ~PyOperationBuilderGuard() { opCls.attr("create") = oldBuilder; }

  py::object opCls;
  py::object oldBuilder;
};

PYBIND11_MODULE(_byteir, m) {
  byteirRegisterAllPasses();
  mlirRegisterAllMhloPasses();

  m.doc() = "byteir python extension";

  //========== Register Dialects ============
  m.def(
      "register_cat_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__cat__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_ace_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__ace__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_ccl_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__ccl__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_byre_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__byre__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_byre_serial_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__byre_serial__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def(
      "register_dialect_extensions",
      [](MlirContext context) { byteirRegisterDialectExtensions(context); },
      py::arg("context"));

  m.def(
      "register_translation_dialects",
      [](MlirContext context) { byteirRegisterTranslationDialects(context); },
      py::arg("context"));

  //============ Translate ==============
  m.def(
      "translate_to_ptx",
      [](MlirModule module, const std::string &ptxPrefixFileName,
         const std::string &gpuArch) {
        if (!byteirTranslateToPTX(module, toMlirStringRef(ptxPrefixFileName),
                                  toMlirStringRef(gpuArch))) {
          PyErr_SetString(PyExc_ValueError, "failed to translate to ptx");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("ptx_prefix_file_name"),
      py::arg("gpu_arch") = "sm_70");
  m.def(
      "translate_to_llvmbc",
      [](MlirModule module, const std::string &outputFile) {
        if (!byteirTranslateToLLVMBC(module, toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError,
                          "failed to translate to llvm bytecode");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("output_file"));
  m.def(
      "translate_to_llvmir",
      [](MlirModule module, const std::string &outputFile) {
        if (!byteirTranslateToLLVMIR(module, toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError, "failed to translate to llvm ir");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("output_file"));

  //============ Byre Serialization ==============
  m.def(
      "serialize_byre",
      [](MlirModule module, const std::string &targetVersion,
         const std::string &outputFile) {
        if (!byteirSerializeByre(module, toMlirStringRef(targetVersion),
                                 toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError, "failed to serialize byre");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("target_version"), py::arg("output_file"));
  m.def("deserialize_byre",
        [](const std::string &artifactStr, MlirContext context) -> MlirModule {
          auto module =
              byteirDeserializeByre(toMlirStringRef(artifactStr), context);
          if (mlirModuleIsNull(module)) {
            PyErr_SetString(PyExc_ValueError, "failed to deserialize byre");
          }
          return module;
        });

  //============ Module Utils ==============
  m.def(
      "merge_two_modules",
      [](MlirModule module0, MlirModule module1,
         std::vector<int64_t> mapping) -> MlirModule {
        auto module = byteirMergeTwoModules(module0, module1, mapping.data(),
                                            mapping.size());
        if (mlirModuleIsNull(module)) {
          PyErr_SetString(PyExc_ValueError, "failed to merge two modules");
          return {};
        }
        return module;
      },
      py::arg("module0"), py::arg("module1"),
      py::arg("mapping") = std::vector<int64_t>{});

  m.def(
      "register_pdl_constraint_fn",
      [](MlirContext ctx, std::string name, std::function<bool(py::args)> py_fn,
         bool override) -> bool {
        std::function<bool(std::vector<MlirPDLValue>)> fn =
            [py_fn](std::vector<MlirPDLValue> pdlValues) {
              py::gil_scoped_acquire _;
              py::tuple args(pdlValues.size());
              for (size_t i = 0; i < pdlValues.size(); ++i) {
                args[i] = wrap(pdlValues[i]);
              }
              return py_fn(args);
            };
        return mlirRegisterPDLConstraintFn(
            ctx, mlirStringRefCreate(name.c_str(), name.size()),
            reinterpret_cast<void *>(&fn), override);
      },
      py::arg("context"), py::arg("name"), py::arg("fn"),
      py::arg("override") = true);

  m.def(
      "register_pdl_rewrite_fn",
      [](MlirContext ctx, std::string name,
         std::function<py::object(py::args)> py_fn,
         std::vector<MlirPDLValueKind> result_tys, bool override) -> bool {
        std::function<bool(MlirOperation, MlirPDLResultListRef,
                           std::vector<MlirPDLValue>,
                           std::function<void(MlirOperation)>)>
            fn = [py_fn, result_tys](
                     MlirOperation insertionPoint,
                     MlirPDLResultListRef pdlResults,
                     std::vector<MlirPDLValue> pdlValues,
                     std::function<void(MlirOperation)> opInsertionCallback) {
              py::gil_scoped_acquire _;
              py::tuple args(pdlValues.size());
              for (size_t i = 0; i < pdlValues.size(); ++i) {
                args[i] = wrap(pdlValues[i]);
              }

              py::object pyIp;
              if (insertionPoint.ptr) {
                pyIp = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                           .attr("InsertionPoint")(insertionPoint);
              }

              py::object pyLoc;
              try {
                py::object curLoc =
                    py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                        .attr("Location")
                        .attr("current");

                if (!curLoc)
                  throw py::value_error("no current location");
              } catch (std::exception &) {
                pyLoc = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                            .attr("Location")
                            .attr("unknown")();
              }

              // TODO: make following hooks as thread local
              PyCtxGuard pyIpGuard(std::move(pyIp));
              PyCtxGuard pyLocGuard(std::move(pyLoc));
              // TODO: check whether re-entering
              PyOpViewBuilderGuard pyOpViewBuilderGuard(opInsertionCallback);
              PyOperationBuilderGuard pyOperationBuilderGuard(
                  opInsertionCallback);

              auto ret = py_fn(args);

              if (result_tys.size() == 1) {
                emplaceResult(pdlResults, ret, result_tys[0]);
              } else {
                auto results = ret.cast<py::sequence>();
                if (results.size() != result_tys.size())
                  return false;

                for (size_t i = 0; i < results.size(); ++i) {
                  emplaceResult(pdlResults, results[i], result_tys[i]);
                }
              }

              return true;
            };
        return mlirRegisterPDLRewriteFn(
            ctx, mlirStringRefCreate(name.c_str(), name.size()),
            reinterpret_cast<void *>(&fn), override);
      },
      py::arg("context"), py::arg("name"), py::arg("fn"),
      py::arg("result_types"), py::arg("override") = true);

  py::enum_<MlirPDLValueKind>(m, "PDLValueKind")
      .value("Attribute", MlirPDLValueAttribute)
      .value("Operation", MlirPDLValueOperation)
      .value("Type", MlirPDLValueType)
      .value("TypeRange", MlirPDLValueTypeRange)
      .value("Value", MlirPDLValueValue)
      .value("ValueRange", MlirPDLValueValueRange);
}
