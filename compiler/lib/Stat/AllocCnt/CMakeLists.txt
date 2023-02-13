add_byteir_stat_library(ByteIRAllocCntStat
  AllocCnt.cpp

  DEPENDS
  ByteIRStatCommon
  ByteIRUtils

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  ByteIRStatCommon
  ByteIRUtils
  MLIRIR
  MLIRMemRefDialect
)