target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compile.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/dtype.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/dtype_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/einsum.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/graph_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/metal.h)

# Define MLX_VERSION only in the version.cpp file.
add_library(mlx_version OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp)
target_compile_definitions(mlx_version PRIVATE MLX_VERSION="${MLX_VERSION}")
# On Windows shared lib builds, mlx_version also needs MLX_EXPORT for proper DLL
# linkage
if(WIN32 AND BUILD_SHARED_LIBS)
  target_compile_definitions(mlx_version PRIVATE MLX_EXPORT)
endif()
target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:mlx_version>)

if(MSVC)
  # Disable some MSVC warnings to speed up compilation. For C++ files, pass
  # flags directly; for CUDA files, wrap with -Xcompiler
  target_compile_options(
    mlx
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4068
           /wd4244
           /wd4267
           /wd4804>
           $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4068
           -Xcompiler=/wd4244
           -Xcompiler=/wd4267
           -Xcompiler=/wd4804>)
endif()

if(WIN32)
  # On Windows, we use explicit __declspec(dllexport) via MLX_API macro instead
  # of WINDOWS_EXPORT_ALL_SYMBOLS to avoid exceeding the 65535 symbol export
  # limit (CUDA template instantiations generate 100k+ symbols).
  set_target_properties(mlx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/common)

if(MLX_BUILD_CPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_cpu)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/distributed)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/io)

if(MLX_BUILD_METAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/metal)
else()
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/no_metal.cpp)
endif()

if(MLX_BUILD_CUDA)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda)
else()
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/no_cuda.cpp)
endif()

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/gpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_gpu)
endif()
