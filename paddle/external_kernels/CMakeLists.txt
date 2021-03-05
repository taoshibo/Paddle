set(external_kernels_src "basic_kernels.cc" "paddle.cc" "register.cc")

include_directories("${PADDLE_BINARY_DIR}/third_party/cinn/cinn/include")
include_directories("${PADDLE_BINARY_DIR}/third_party/cinn/third_party/llvm11/include")
cc_library(external_kernels SHARED SRCS ${external_kernels_src})

#target_link_libraries(external_kernels ${PADDLE_BINARY_DIR}/third_party/cinn/cinn/lib/libcinncore.a)
#target_link_libraries(external_kernels ${PADDLE_BINARY_DIR}/third_party/cinn/cinn/lib/libframework_proto.a)

set_target_properties(external_kernels PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
target_link_libraries(external_kernels ${fluid_modules})

#file(GLOB LLVM_LIBS ${PADDLE_BINARY_DIR}/third_party/cinn/third_party/llvm11/lib/*.a)
#target_link_libraries(external_kernels ${LLVM_LIBS})
#file(GLOB FRAMEWORK_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/framework/*.a)
#target_link_libraries(external_kernels ${FRAMEWORK_LIBS})
#message(STATUS "FRAMEWORK_LIBS1: ${FRAMEWORK_LIBS}")
#file(GLOB FRAMEWORK_DETAILS_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/framework/details/*.a)
#target_link_libraries(external_kernels ${FRAMEWORK_DETAILS_LIBS})
#file(GLOB FRAMEWORK_IR_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/framework/ir/*.a)
#target_link_libraries(external_kernels ${FRAMEWORK_IR_LIBS})
#file(GLOB PLATFORM_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/platform/*.a)
#target_link_libraries(external_kernels ${PLATFORM_LIBS})
#file(GLOB ALLOCATION_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/memory/allocation/*.a)
#target_link_libraries(external_kernels ${ALLOCATION_LIBS})
#file(GLOB MEMORY_LIBS ${PADDLE_BINARY_DIR}/paddle/fluid/memory/*.a)
#target_link_libraries(external_kernels ${MEMORY_LIBS})
#target_link_libraries(external_kernels ${PADDLE_BINARY_DIR}/paddle/fluid/framework/libdata_type.a)

