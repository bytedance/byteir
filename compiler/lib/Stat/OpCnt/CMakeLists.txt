add_byteir_stat_library(ByteIROpCntStat
  OpCnt.cpp

  DEPENDS
  ByteIRStatCommon

  LINK_LIBS PUBLIC
  ByteIRStatCommon
  MLIRIR
)