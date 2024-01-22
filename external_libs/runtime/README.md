# Runtime External Libs

Runtime external library contains standalone kernels that can be used externally, eg. used by ByteIR Runtime.

## Build
### Linux/Mac
```bash
mkdir ./build

# build runtime
cd build && cmake .. -G Ninja

cmake --build . --target all
```
