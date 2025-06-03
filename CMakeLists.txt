cmake_minimum_required(VERSION 3.16)
project(harmonics LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(micro_gtest_main STATIC third_party/gtest/gtest_main.cpp)
target_include_directories(micro_gtest_main PUBLIC third_party/gtest/include)
add_library(micro_gtest INTERFACE)
target_include_directories(micro_gtest INTERFACE third_party/gtest/include)
add_library(GTest::gtest ALIAS micro_gtest)
add_library(GTest::gtest_main ALIAS micro_gtest_main)

option(ENABLE_UNITY_BUILD "Enable CMake unity build to speed up compilation" ON)
if(ENABLE_UNITY_BUILD)
    set(CMAKE_UNITY_BUILD ON)
    message(STATUS "Unity build enabled")
endif()

# Optional LTO/PGO support
option(ENABLE_LTO "Enable link time optimisation" OFF)
set(PGO_PHASE "OFF" CACHE STRING "Profile guided optimisation phase (OFF|GENERATE|USE)")
set_property(CACHE PGO_PHASE PROPERTY STRINGS OFF GENERATE USE)

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_output)
    if(lto_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO enabled")
    else()
        message(WARNING "LTO not supported: ${lto_output}")
    endif()
endif()

if(PGO_PHASE STREQUAL "GENERATE")
    message(STATUS "PGO instrumentation enabled")
    add_compile_options(-fprofile-generate -fprofile-dir=${CMAKE_BINARY_DIR}/profiles)
    add_link_options(-fprofile-generate -fprofile-dir=${CMAKE_BINARY_DIR}/profiles)
elseif(PGO_PHASE STREQUAL "USE")
    message(STATUS "PGO optimization enabled")
    add_compile_options(-fprofile-use -fprofile-dir=${CMAKE_BINARY_DIR}/profiles -fprofile-correction)
    add_link_options(-fprofile-use -fprofile-dir=${CMAKE_BINARY_DIR}/profiles -fprofile-correction)
endif()

# Optional Apache Arrow support
option(HARMONICS_HAS_ARROW "Enable Apache Arrow integration" OFF)

find_package(Vulkan QUIET)

# Workaround for pthreads detection issues on macOS/Apple Silicon
if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

# Find Vulkan if available
if(Vulkan_FOUND)
    message(STATUS "Found Vulkan: ${Vulkan_INCLUDE_DIRS}")
else()
    message(STATUS "Vulkan not found, some features may be disabled.")
endif()

# --------------------------------------------------------------------
# Protobuf and gRPC setup
# --------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(zstd REQUIRED)
if(NOT Protobuf_FOUND)
    message(FATAL_ERROR "Could not find Protobuf.")
endif()
if(NOT gRPC_FOUND)
    message(FATAL_ERROR "Could not find gRPC.")
endif()
find_program(gRPC_CPP_PLUGIN_EXECUTABLE NAMES grpc_cpp_plugin REQUIRED)
set(PROTO_GRPC_SRCS)
set(PROTO_GRPC_HDRS)
file(GLOB PROTO_FILES
    "${CMAKE_SOURCE_DIR}/third_party/onnx_tiny/*.proto"
    "${CMAKE_SOURCE_DIR}/protos/*.proto")
message(STATUS "PROTO_FILES = ${PROTO_FILES}")
foreach(proto_file ${PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    get_filename_component(proto_path ${proto_file} PATH)
    set(proto_cc "${proto_path}/${proto_name}.pb.cc")
    set(proto_h  "${proto_path}/${proto_name}.pb.h")
    set(grpc_cc  "${proto_path}/${proto_name}.grpc.pb.cc")
    set(grpc_h   "${proto_path}/${proto_name}.grpc.pb.h")
    add_custom_command(
        OUTPUT ${proto_cc} ${proto_h}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${proto_path}
             -I ${proto_path}
             ${proto_file}
        DEPENDS ${proto_file}
    )
    add_custom_command(
        OUTPUT ${grpc_cc} ${grpc_h}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --grpc_out=${proto_path}
             --plugin=protoc-gen-grpc=${gRPC_CPP_PLUGIN_EXECUTABLE}
             -I ${proto_path}
             ${proto_file}
        DEPENDS ${proto_file}
    )
    list(APPEND PROTO_SRCS ${proto_cc} ${grpc_cc})
    list(APPEND PROTO_HDRS ${proto_h} ${grpc_h})
endforeach()
add_custom_target(generate_protos DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
find_program(gRPC_CPP_PLUGIN_EXECUTABLE NAMES grpc_cpp_plugin REQUIRED)

# ──────────────────────────────────────────────────────────────
# **BLAKE3** – DO NOT MODIFY THIS SECTION! You will break the build!
# ──────────────────────────────────────────────────────────────
set(BLAKE3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/BLAKE3)
set(BLAKE3_SOURCES
    ${BLAKE3_DIR}/c/blake3.c
    ${BLAKE3_DIR}/c/blake3_dispatch.c
    ${BLAKE3_DIR}/c/blake3_portable.c
)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND BLAKE3_SOURCES
        ${BLAKE3_DIR}/c/blake3_sse2_x86-64_unix.S
        ${BLAKE3_DIR}/c/blake3_sse41_x86-64_unix.S
        ${BLAKE3_DIR}/c/blake3_avx2_x86-64_unix.S
        ${BLAKE3_DIR}/c/blake3_avx512_x86-64_unix.S
    )
    set_source_files_properties(${BLAKE3_DIR}/c/blake3_avx512_x86-64_unix.S
        PROPERTIES COMPILE_FLAGS "-mavx512f")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    list(APPEND BLAKE3_SOURCES
        ${BLAKE3_DIR}/c/blake3_neon.c)
endif()
add_library(blake3 STATIC ${BLAKE3_SOURCES})
target_include_directories(blake3 PUBLIC ${BLAKE3_DIR}/c)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_definitions(blake3 PUBLIC BLAKE3_USE_NEON)
else()
    target_compile_definitions(blake3 PUBLIC BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512 BLAKE3_NO_NEON)
endif()
add_library(BLAKE3::blake3 ALIAS blake3)

# ---------------------------------------------------------------------
# Compile shaders into header
# ---------------------------------------------------------------------
file(GLOB SHADER_COMP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/Shaders.hpp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile_shaders.sh
    DEPENDS ${SHADER_COMP_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(generate_shaders DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/Shaders.hpp)

# ──────────────────────────────────────────────────────────────
# Harmonics library setup
# ──────────────────────────────────────────────────────────────
if(Protobuf_FOUND)
    add_library(onnx_tiny STATIC third_party/onnx_tiny/onnx_tiny.pb.cc)
    target_include_directories(onnx_tiny PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx_tiny
        ${protobuf_INCLUDE_DIRS}
        ${gRPC_INCLUDE_DIRS}
    )
    target_link_libraries(onnx_tiny PUBLIC protobuf::libprotobuf)
    add_library(tensor_stream_proto STATIC
        protos/tensor_stream.pb.cc
        protos/tensor_stream.grpc.pb.cc)
    target_include_directories(tensor_stream_proto PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/protos
        ${protobuf_INCLUDE_DIRS}
        ${gRPC_INCLUDE_DIRS})
    target_link_libraries(tensor_stream_proto PUBLIC protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)
endif()
add_library(harmonics INTERFACE)
add_dependencies(harmonics generate_shaders)
target_include_directories(harmonics INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/protos>
    ${protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)
if(HARMONICS_HAS_ARROW)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/arrow")
        target_include_directories(harmonics INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/arrow/cpp/src)
        target_compile_definitions(harmonics INTERFACE HARMONICS_HAS_ARROW=1)
    else()
        message(WARNING "Arrow support requested but sources not found")
    endif()
endif()
if(Protobuf_FOUND)
    target_include_directories(harmonics INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx_tiny>
        ${protobuf_INCLUDE_DIRS}
        ${gRPC_INCLUDE_DIRS})
endif()
target_link_libraries(harmonics INTERFACE
    Threads::Threads
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc++_reflection
    tensor_stream_proto
    BLAKE3::blake3
    zstd::libzstd_static
)
if(WIN32)
    target_link_libraries(harmonics INTERFACE ws2_32)
endif()
if(Vulkan_FOUND)
    target_link_libraries(harmonics INTERFACE Vulkan::Vulkan)
endif()

option(HARMONICS_HAS_JS "Build JavaScript bindings" OFF)
option(HARMONICS_HAS_JNI "Build JNI bindings" OFF)
if(HARMONICS_HAS_JS)
    find_package(V8 REQUIRED)
    add_library(harmonics_js SHARED src/javascript_bindings.cpp)
    target_include_directories(harmonics_js PUBLIC include)
    target_link_libraries(harmonics_js PUBLIC harmonics V8::V8)
endif()
if(HARMONICS_HAS_JNI)
    find_package(JNI REQUIRED)
    add_library(harmonics_jni SHARED src/jni_bindings.cpp)
    target_include_directories(harmonics_jni PUBLIC include ${JNI_INCLUDE_DIRS})
    target_link_libraries(harmonics_jni PUBLIC harmonics ${JNI_LIBRARIES})
endif()

option(HARMONICS_HAS_PY "Build Python bindings" OFF)
if(HARMONICS_HAS_PY)
    find_package(Python3 COMPONENTS Development REQUIRED)
    add_library(harmonics_py MODULE src/python_bindings.cpp)
    target_include_directories(harmonics_py PUBLIC include)
    target_link_libraries(harmonics_py PUBLIC harmonics Python3::Python)
    target_compile_definitions(harmonics_py PRIVATE HARMONICS_HAS_PY=1)
    set_target_properties(harmonics_py PROPERTIES PREFIX "")
endif()

add_library(harmonics_ffi STATIC src/rust_ffi.cpp)
target_include_directories(harmonics_ffi PUBLIC include)
target_link_libraries(harmonics_ffi PUBLIC harmonics)

# Optional hardware-dependent tests
option(ENABLE_HARDWARE_TESTS "Build tests requiring GPUs or other hardware" OFF)

# Command line tool -----------------------------------------------------------
add_executable(harmonics_cli src/harmonics.cpp)
target_link_libraries(harmonics_cli PRIVATE harmonics BLAKE3::blake3)

# Graph editing command line tool
add_executable(graph_cli src/graph_cli.cpp)
target_link_libraries(graph_cli PRIVATE harmonics BLAKE3::blake3)

# Graph diff and merge command line tool
add_executable(graph_diff src/graph_diff_tool.cpp)
target_link_libraries(graph_diff PRIVATE harmonics BLAKE3::blake3)

# Graph debugging visualizer
add_executable(graph_debugger src/graph_debugger.cpp)
target_link_libraries(graph_debugger PRIVATE harmonics BLAKE3::blake3)

# Graph inspection command line tool
add_executable(graph_info src/graph_info_cli.cpp)
target_link_libraries(graph_info PRIVATE harmonics BLAKE3::blake3)

if(Protobuf_FOUND)
    add_executable(model_convert src/model_convert.cpp)
    target_link_libraries(model_convert PRIVATE harmonics BLAKE3::blake3 onnx_tiny)
endif()

add_executable(dataset_convert src/dataset_convert.cpp)
target_link_libraries(dataset_convert PRIVATE harmonics BLAKE3::blake3)

# CLI for transferring cached datasets over the network
add_executable(dataset_cache_cli src/dataset_cache_cli.cpp)
target_link_libraries(dataset_cache_cli PRIVATE harmonics BLAKE3::blake3)

# CLI for validating dataset schemas
add_executable(dataset_schema_cli src/dataset_schema_cli.cpp)
target_link_libraries(dataset_schema_cli PRIVATE harmonics BLAKE3::blake3)

# CLI for packaging and installing plugins
add_executable(plugin_packager src/plugin_packager.cpp)
target_link_libraries(plugin_packager PRIVATE harmonics BLAKE3::blake3)

add_subdirectory(benchmarks)
# -------------------------------------------------------------------------
# Examples and tests
# -------------------------------------------------------------------------
add_subdirectory(examples)
enable_testing()
add_subdirectory(tests)
