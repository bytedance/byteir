// RUN: onnx-frontend %S/dynamic_shape_relu.onnx --input-name-and-shapes=X,1,128,80 -- | FileCheck %s

// CHECK: func.func @main(%arg0: tensor<1x128x80xf32>)