//===- CppEmitter.h - C++ emitter ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines C++ emitter code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#ifndef BYTEIR_TARGET_CPP_CPPEMITTER_H
#define BYTEIR_TARGET_CPP_CPPEMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace byteir {

class CppEmitter {
public:
  explicit CppEmitter(llvm::raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  virtual mlir::LogicalResult emitAttribute(mlir::Location loc,
                                            mlir::Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  virtual mlir::LogicalResult emitOperation(mlir::Operation &op,
                                            bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  virtual mlir::LogicalResult emitType(mlir::Location loc, mlir::Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  virtual mlir::LogicalResult emitTypes(mlir::Location loc,
                                        mlir::ArrayRef<mlir::Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  virtual mlir::LogicalResult emitTupleType(mlir::Location loc,
                                            mlir::ArrayRef<mlir::Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  virtual mlir::LogicalResult emitVariableAssignment(mlir::OpResult result);

  /// Emits a variable declaration for a result of an operation.
  virtual mlir::LogicalResult emitVariableDeclaration(mlir::OpResult result,
                                                      bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  virtual mlir::LogicalResult emitAssignPrefix(mlir::Operation &op);

  /// Emits a label for the block.
  virtual mlir::LogicalResult emitLabel(mlir::Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  virtual mlir::LogicalResult
  emitOperandsAndAttributes(mlir::Operation &op,
                            mlir::ArrayRef<mlir::StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  virtual mlir::LogicalResult emitOperands(mlir::Operation &op);

  /// Return the existing or a new name for a Value.
  virtual llvm::StringRef getOrCreateName(mlir::Value val);

  /// Return the existing or a new label of a Block.
  virtual llvm::StringRef getOrCreateName(mlir::Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(mlir::IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<mlir::Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<mlir::Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(mlir::Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(mlir::Block &block);

  /// Returns the output stream.
  mlir::raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

protected:
  using ValueMapper = llvm::ScopedHashTable<mlir::Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<mlir::Block *, std::string>;

  /// Output stream to emit to.
  mlir::raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};

} // namespace byteir

#endif // BYTEIR_TARGET_CPP_CPPEMITTER_H
