# Accept ONNXRUNTIME_ROOTDIR (CMakePresets), onnxruntime_DIR, or ONNXRUNTIME_DIR env var
if(NOT ONNXRUNTIME_ROOTDIR)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_ROOTDIR "$ENV{ONNXRUNTIME_DIR}")
    endif()
endif()

# Normalize path to remove any '..' components that confuse cmake's install interface checks
if(ONNXRUNTIME_ROOTDIR)
    get_filename_component(ONNXRUNTIME_ROOTDIR "${ONNXRUNTIME_ROOTDIR}" REALPATH)
endif()

set(onnxruntime_INCLUDE_DIR ${ONNXRUNTIME_ROOTDIR}/include)
set(onnxruntime_LIBRARY ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so)

# check if the ONNX Runtime library exists
if(NOT EXISTS ${onnxruntime_LIBRARY})
    message(FATAL_ERROR "ONNX Runtime library not found at ${onnxruntime_LIBRARY}")
    set(onnxruntime_FOUND FALSE)
endif()