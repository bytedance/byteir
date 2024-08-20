add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Pipelines)
add_subdirectory(Stat)
add_subdirectory(Target)
add_subdirectory(Transforms)
add_subdirectory(Utils)

# note: CAPI depends on byteir property, so add it at last
add_subdirectory(CAPI)
