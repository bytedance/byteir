#ifndef BYTEIR_DIALECT_CCL_CCLOP_INTERFACE
#define BYTEIR_DIALECT_CCL_CCLOP_INTERFACE
include "mlir/IR/OpBase.td"

def CclSynchronousOpInterface : OpInterface<"CclSynchronousOpInterface"> {
  let cppNamespace = "mlir::ccl";
  let methods = [
    InterfaceMethod<
        /*desc=*/[{ 
          Returns the value of synchronous in ccl operation. 
        }],
        /*retType=*/"bool",
        /*methodName=*/"getSynchronous",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return getSynchronous();
        }]
      >,
    InterfaceMethod<
        /*desc=*/[{ Returns a string for synchronous in a ccl operation. }],
        /*retType=*/"StringRef",
        /*methodName=*/"getSynchronousName",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ 
          return "synchronous"; 
        }]
      >
  ];
}

#endif // BYTEIR_DIALECT_CCL_CCLOP_INTERFACE
