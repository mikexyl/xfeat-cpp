option(onnxruntime_DIR "Path to ONNX Runtime root directory" "")

set(onnxruntime_INCLUDE_DIR ${onnxruntime_DIR}/include)
set(onnxruntime_LIBRARY ${onnxruntime_DIR}/lib/libonnxruntime.so)

# check if the ONNX Runtime library exists
if(NOT EXISTS ${onnxruntime_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime library not found at ${onnxruntime_LIBRARY}")
    set(onnxruntime_FOUND FALSE)
endif()