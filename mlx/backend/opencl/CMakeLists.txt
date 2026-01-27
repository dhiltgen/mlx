# OpenCL backend for MLX

# Find OpenCL
find_package(OpenCL REQUIRED)

# OpenCL backend sources
set(OPENCL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/arange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/available.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
)

target_sources(mlx PRIVATE ${OPENCL_SOURCES})
target_include_directories(mlx PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(mlx PRIVATE ${OpenCL_LIBRARIES})

# Add OpenCL compile definition
target_compile_definitions(mlx PRIVATE MLX_BUILD_OPENCL)

message(STATUS "OpenCL backend enabled")
message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
