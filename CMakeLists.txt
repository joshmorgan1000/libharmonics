cmake_minimum_required(VERSION 3.16)
project(harmonics LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_UNITY_BUILD "Enable CMake unity build to speed up compilation" OFF)
if(ENABLE_UNITY_BUILD)
    set(CMAKE_UNITY_BUILD ON)
    message(STATUS "Unity build enabled")
endif()

enable_testing()
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
file(GLOB PROTO_FILES "${CMAKE_SOURCE_DIR}/third_party/onnx_tiny/*.proto")
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
endif()
add_library(harmonics INTERFACE)
add_dependencies(harmonics generate_shaders)
target_include_directories(harmonics INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)
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
if(HARMONICS_HAS_JS)
    find_package(V8 REQUIRED)
    add_library(harmonics_js SHARED src/javascript_bindings.cpp)
    target_include_directories(harmonics_js PUBLIC include)
    target_link_libraries(harmonics_js PUBLIC harmonics V8::V8)
endif()

add_library(harmonics_ffi STATIC src/rust_ffi.cpp)
target_include_directories(harmonics_ffi PUBLIC include)
target_link_libraries(harmonics_ffi PUBLIC harmonics)

# Optional hardware-dependent tests
option(ENABLE_HARDWARE_TESTS "Build tests requiring GPUs or other hardware" OFF)

# ---------------------------------------------------------------------
# Test and example executables
# ---------------------------------------------------------------------
add_executable(multi_gpu_training examples/multi_gpu_training.cpp)
target_include_directories(multi_gpu_training PRIVATE include)
target_link_libraries(multi_gpu_training PRIVATE harmonics BLAKE3::blake3)

add_executable(dummy_test tests/dummy_test.cpp)
target_include_directories(dummy_test PRIVATE tests)
target_link_libraries(dummy_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME dummy_test COMMAND dummy_test)

add_executable(function_registry_test tests/function_registry_test.cpp)
target_include_directories(function_registry_test PRIVATE tests)
target_link_libraries(function_registry_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME function_registry_test COMMAND function_registry_test)

add_executable(function_registry_thread_test tests/function_registry_thread_test.cpp)
target_include_directories(function_registry_thread_test PRIVATE tests)
target_link_libraries(function_registry_thread_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME function_registry_thread_test COMMAND function_registry_thread_test)

add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_include_directories(tokenizer_test PRIVATE tests)
target_link_libraries(tokenizer_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(parser_test tests/parser_test.cpp)
target_include_directories(parser_test PRIVATE tests)
target_link_libraries(parser_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME parser_test COMMAND parser_test)

add_executable(graph_test tests/graph_test.cpp)
target_include_directories(graph_test PRIVATE tests)
target_link_libraries(graph_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME graph_test COMMAND graph_test)

add_executable(cycle_runtime_test tests/cycle_runtime_test.cpp)
target_include_directories(cycle_runtime_test PRIVATE tests)
target_link_libraries(cycle_runtime_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME cycle_runtime_test COMMAND cycle_runtime_test)

add_executable(threaded_runtime_test tests/threaded_runtime_test.cpp)
target_include_directories(threaded_runtime_test PRIVATE tests)
target_link_libraries(threaded_runtime_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME threaded_runtime_test COMMAND threaded_runtime_test)

add_executable(training_tap_test tests/training_tap_test.cpp)
target_include_directories(training_tap_test PRIVATE tests)
target_link_libraries(training_tap_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME training_tap_test COMMAND training_tap_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(runtime_api_test tests/runtime_api_test.cpp)
    target_include_directories(runtime_api_test PRIVATE tests)
    target_link_libraries(runtime_api_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME runtime_api_test COMMAND runtime_api_test)
endif()

add_executable(mnist_example_test tests/mnist_example_test.cpp)
target_include_directories(mnist_example_test PRIVATE tests)
target_link_libraries(mnist_example_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME mnist_example_test COMMAND mnist_example_test)

add_executable(mnist_training_cycle_test tests/mnist_training_cycle_test.cpp)
target_include_directories(mnist_training_cycle_test PRIVATE tests)
target_link_libraries(mnist_training_cycle_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME mnist_training_cycle_test COMMAND mnist_training_cycle_test)

add_executable(secure_mode_test tests/secure_mode_test.cpp)
target_include_directories(secure_mode_test PRIVATE tests
    ${protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS})
target_link_libraries(secure_mode_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME secure_mode_test COMMAND secure_mode_test)

add_executable(gpu_backend_test tests/gpu_backend_test.cpp)
target_include_directories(gpu_backend_test PRIVATE tests)
target_link_libraries(gpu_backend_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME gpu_backend_test COMMAND gpu_backend_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(vulkan_device_test tests/vulkan_device_test.cpp)
    target_include_directories(vulkan_device_test PRIVATE tests)
    target_link_libraries(vulkan_device_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME vulkan_device_test COMMAND vulkan_device_test)

    add_executable(cuda_device_test tests/cuda_device_test.cpp)
    target_include_directories(cuda_device_test PRIVATE tests)
    target_link_libraries(cuda_device_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME cuda_device_test COMMAND cuda_device_test)
endif()

add_executable(opencl_device_test tests/opencl_device_test.cpp)
target_include_directories(opencl_device_test PRIVATE tests)
target_link_libraries(opencl_device_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME opencl_device_test COMMAND opencl_device_test)

if(Vulkan_FOUND)
    add_executable(vulkan_backend_test tests/vulkan_backend_test.cpp)
    target_include_directories(vulkan_backend_test PRIVATE tests)
    target_link_libraries(vulkan_backend_test PRIVATE harmonics BLAKE3::blake3 Vulkan::Vulkan)
    add_test(NAME vulkan_backend_test COMMAND vulkan_backend_test)
endif()

if(ENABLE_HARDWARE_TESTS)
    add_executable(vulkan_backend_enabled_test tests/vulkan_backend_enabled_test.cpp)
    target_include_directories(vulkan_backend_enabled_test PRIVATE tests)
    target_link_libraries(vulkan_backend_enabled_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME vulkan_backend_enabled_test COMMAND vulkan_backend_enabled_test)
endif()

add_executable(cuda_backend_enabled_test tests/cuda_backend_enabled_test.cpp)
target_include_directories(cuda_backend_enabled_test PRIVATE tests)
target_link_libraries(cuda_backend_enabled_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME cuda_backend_enabled_test COMMAND cuda_backend_enabled_test)

add_executable(gpu_fallback_test tests/gpu_fallback_test.cpp)
target_include_directories(gpu_fallback_test PRIVATE tests)
target_link_libraries(gpu_fallback_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME gpu_fallback_test COMMAND gpu_fallback_test)

add_executable(gpu_shader_wrapper_test tests/gpu_shader_wrapper_test.cpp)
target_include_directories(gpu_shader_wrapper_test PRIVATE tests)
target_link_libraries(gpu_shader_wrapper_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME gpu_shader_wrapper_test COMMAND gpu_shader_wrapper_test)

add_executable(int8_matmul_test tests/int8_matmul_test.cpp)
target_include_directories(int8_matmul_test PRIVATE tests)
target_link_libraries(int8_matmul_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME int8_matmul_test COMMAND int8_matmul_test)

add_executable(int8_determinism_test tests/int8_determinism_test.cpp)
target_include_directories(int8_determinism_test PRIVATE tests)
target_link_libraries(int8_determinism_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME int8_determinism_test COMMAND int8_determinism_test)

add_executable(int_sgd_update_test tests/int_sgd_update_test.cpp)
target_include_directories(int_sgd_update_test PRIVATE tests)
target_link_libraries(int_sgd_update_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME int_sgd_update_test COMMAND int_sgd_update_test)

add_executable(int4_matmul_test tests/int4_matmul_test.cpp)
target_include_directories(int4_matmul_test PRIVATE tests)
target_link_libraries(int4_matmul_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME int4_matmul_test COMMAND int4_matmul_test)

add_executable(int3_matmul_test tests/int3_matmul_test.cpp)
target_include_directories(int3_matmul_test PRIVATE tests)
target_link_libraries(int3_matmul_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME int3_matmul_test COMMAND int3_matmul_test)

add_executable(gpu_kernel_api_test tests/gpu_kernel_api_test.cpp)
target_include_directories(gpu_kernel_api_test PRIVATE tests)
target_link_libraries(gpu_kernel_api_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME gpu_kernel_api_test COMMAND gpu_kernel_api_test)

add_executable(wrapper_ring_buffer_test tests/wrapper_ring_buffer_test.cpp)
target_include_directories(wrapper_ring_buffer_test PRIVATE tests)
target_link_libraries(wrapper_ring_buffer_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME wrapper_ring_buffer_test COMMAND wrapper_ring_buffer_test)

add_executable(accelerator_selection_cpu_test tests/accelerator_selection_cpu_test.cpp)
target_include_directories(accelerator_selection_cpu_test PRIVATE tests)
target_link_libraries(accelerator_selection_cpu_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME accelerator_selection_cpu_test COMMAND accelerator_selection_cpu_test)

add_executable(accelerator_selection_gpu_test tests/accelerator_selection_gpu_test.cpp)
target_include_directories(accelerator_selection_gpu_test PRIVATE tests)
target_link_libraries(accelerator_selection_gpu_test PRIVATE harmonics BLAKE3::blake3 dl)
add_test(NAME accelerator_selection_gpu_test COMMAND accelerator_selection_gpu_test)

add_executable(fpga_fallback_test tests/fpga_fallback_test.cpp)
target_include_directories(fpga_fallback_test PRIVATE tests)
target_link_libraries(fpga_fallback_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME fpga_fallback_test COMMAND fpga_fallback_test)

add_executable(gpu_fpga_fallback_integration_test tests/gpu_fpga_fallback_integration_test.cpp)
target_include_directories(gpu_fpga_fallback_integration_test PRIVATE tests)
target_link_libraries(gpu_fpga_fallback_integration_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME gpu_fpga_fallback_integration_test COMMAND gpu_fpga_fallback_integration_test)

add_executable(fpga_backend_test tests/fpga_backend_test.cpp)
target_include_directories(fpga_backend_test PRIVATE tests)
target_link_libraries(fpga_backend_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME fpga_backend_test COMMAND fpga_backend_test)

add_executable(wasm_backend_test tests/wasm_backend_test.cpp)
target_include_directories(wasm_backend_test PRIVATE tests)
target_link_libraries(wasm_backend_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME wasm_backend_test COMMAND wasm_backend_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(accelerator_selection_wasm_test tests/accelerator_selection_wasm_test.cpp)
    target_include_directories(accelerator_selection_wasm_test PRIVATE tests)
    target_link_libraries(accelerator_selection_wasm_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME accelerator_selection_wasm_test COMMAND accelerator_selection_wasm_test)
endif()

add_executable(wasm_api_test tests/wasm_api_test.cpp)
target_include_directories(wasm_api_test PRIVATE tests)
target_link_libraries(wasm_api_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME wasm_api_test COMMAND wasm_api_test)

add_executable(fpga_kernel_compiler_test tests/fpga_kernel_compiler_test.cpp)
target_include_directories(fpga_kernel_compiler_test PRIVATE tests)
target_link_libraries(fpga_kernel_compiler_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME fpga_kernel_compiler_test COMMAND fpga_kernel_compiler_test)

add_executable(precision_utils_test tests/precision_utils_test.cpp)
target_include_directories(precision_utils_test PRIVATE tests)
target_link_libraries(precision_utils_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME precision_utils_test COMMAND precision_utils_test)

add_executable(layer_functions_test tests/layer_functions_test.cpp)
target_include_directories(layer_functions_test PRIVATE tests)
target_link_libraries(layer_functions_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME layer_functions_test COMMAND layer_functions_test)

add_executable(activation_library_test tests/activation_library_test.cpp)
target_include_directories(activation_library_test PRIVATE tests)
target_link_libraries(activation_library_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME activation_library_test COMMAND activation_library_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(dataset_producer_test tests/dataset_producer_test.cpp)
    target_include_directories(dataset_producer_test PRIVATE tests)
    target_link_libraries(dataset_producer_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME dataset_producer_test COMMAND dataset_producer_test)

    add_executable(augmentation_test tests/augmentation_test.cpp)
    target_include_directories(augmentation_test PRIVATE tests)
    target_link_libraries(augmentation_test PRIVATE harmonics BLAKE3::blake3)
    add_test(NAME augmentation_test COMMAND augmentation_test)
endif()

add_executable(async_producer_test tests/async_producer_test.cpp)
target_include_directories(async_producer_test PRIVATE tests)
target_link_libraries(async_producer_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME async_producer_test COMMAND async_producer_test)

if(Protobuf_FOUND)
    add_executable(model_import_test tests/model_import_test.cpp)
    target_include_directories(model_import_test PRIVATE tests)
    target_link_libraries(model_import_test PRIVATE harmonics BLAKE3::blake3 onnx_tiny)
    add_test(NAME model_import_test COMMAND model_import_test)

    add_executable(small_llm_test tests/small_llm_test.cpp)
    target_include_directories(small_llm_test PRIVATE tests)
    target_link_libraries(small_llm_test PRIVATE harmonics BLAKE3::blake3 onnx_tiny)
    add_test(NAME small_llm_test COMMAND small_llm_test)

    add_executable(onnx_export_test tests/onnx_export_test.cpp)
    target_include_directories(onnx_export_test PRIVATE tests)
    target_link_libraries(onnx_export_test PRIVATE harmonics BLAKE3::blake3 onnx_tiny)
    add_test(NAME onnx_export_test COMMAND onnx_export_test)
endif()

add_executable(producer_consumer_io_test tests/producer_consumer_io_test.cpp)
target_include_directories(producer_consumer_io_test PRIVATE tests)
target_link_libraries(producer_consumer_io_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME producer_consumer_io_test COMMAND producer_consumer_io_test)

add_executable(transport_helpers_test tests/transport_helpers_test.cpp)
target_include_directories(transport_helpers_test PRIVATE tests)
target_link_libraries(transport_helpers_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME transport_helpers_test COMMAND transport_helpers_test)

add_executable(tcp_io_test tests/tcp_io_test.cpp)
target_include_directories(tcp_io_test PRIVATE tests)
target_link_libraries(tcp_io_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME tcp_io_test COMMAND tcp_io_test)

add_executable(websocket_io_test tests/websocket_io_test.cpp)
target_include_directories(websocket_io_test PRIVATE tests)
target_link_libraries(websocket_io_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME websocket_io_test COMMAND websocket_io_test)

add_executable(grpc_io_test tests/grpc_io_test.cpp)
target_include_directories(grpc_io_test PRIVATE tests)
target_link_libraries(grpc_io_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME grpc_io_test COMMAND grpc_io_test)

add_executable(distributed_io_test tests/distributed_io_test.cpp)
target_include_directories(distributed_io_test PRIVATE tests)
target_link_libraries(distributed_io_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME distributed_io_test COMMAND distributed_io_test)


add_executable(serialization_test tests/serialization_test.cpp)
target_include_directories(serialization_test PRIVATE tests)
target_link_libraries(serialization_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME serialization_test COMMAND serialization_test)

add_executable(introspection_test tests/introspection_test.cpp)
add_executable(graph_edit_test tests/graph_edit_test.cpp)
target_include_directories(graph_edit_test PRIVATE tests)
target_link_libraries(graph_edit_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME graph_edit_test COMMAND graph_edit_test)

add_executable(graph_diff_test tests/graph_diff_test.cpp)
target_include_directories(graph_diff_test PRIVATE tests)
target_link_libraries(graph_diff_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME graph_diff_test COMMAND graph_diff_test)

add_executable(kernel_cache_test tests/kernel_cache_test.cpp)
target_include_directories(kernel_cache_test PRIVATE tests)
target_link_libraries(kernel_cache_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME kernel_cache_test COMMAND kernel_cache_test)

add_executable(shader_cache_test tests/shader_cache_test.cpp)
target_include_directories(shader_cache_test PRIVATE tests)
target_link_libraries(shader_cache_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME shader_cache_test COMMAND shader_cache_test)
target_include_directories(introspection_test PRIVATE tests)
target_link_libraries(introspection_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME introspection_test COMMAND introspection_test)

add_executable(partition_test tests/partition_test.cpp)
target_include_directories(partition_test PRIVATE tests)
target_link_libraries(partition_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME partition_test COMMAND partition_test)

add_executable(auto_partition_test tests/auto_partition_test.cpp)
target_include_directories(auto_partition_test PRIVATE tests)
target_link_libraries(auto_partition_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME auto_partition_test COMMAND auto_partition_test)

# Quantum stub test -------------------------------------------------
add_executable(quantum_stub_test tests/quantum_stub_test.cpp)
target_include_directories(quantum_stub_test PRIVATE tests)
target_link_libraries(quantum_stub_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME quantum_stub_test COMMAND quantum_stub_test)

# Quantum backend test -------------------------------------------------
add_executable(quantum_backend_test tests/quantum_backend_test.cpp)
target_include_directories(quantum_backend_test PRIVATE tests)
target_link_libraries(quantum_backend_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME quantum_backend_test COMMAND quantum_backend_test)

# Quantum hardware test
add_executable(quantum_hardware_test tests/quantum_hardware_test.cpp)
target_include_directories(quantum_hardware_test PRIVATE tests)
target_link_libraries(quantum_hardware_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME quantum_hardware_test COMMAND quantum_hardware_test)

# Plugin test -------------------------------------------------------
add_library(test_plugin SHARED tests/test_plugin.cpp)
target_link_libraries(test_plugin PRIVATE harmonics BLAKE3::blake3)
set_target_properties(test_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(plugin_test tests/plugin_test.cpp)
target_include_directories(plugin_test PRIVATE tests)
target_link_libraries(plugin_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME plugin_test COMMAND plugin_test)

add_executable(plugin_async_test tests/plugin_async_test.cpp)
target_include_directories(plugin_async_test PRIVATE tests)
target_link_libraries(plugin_async_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME plugin_async_test COMMAND plugin_async_test)

add_executable(plugin_reload_test tests/plugin_reload_test.cpp)
target_include_directories(plugin_reload_test PRIVATE tests)
target_link_libraries(plugin_reload_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME plugin_reload_test COMMAND plugin_reload_test)

add_executable(plugin_packager_test tests/plugin_packager_test.cpp)
target_include_directories(plugin_packager_test PRIVATE tests)
target_link_libraries(plugin_packager_test PRIVATE harmonics BLAKE3::blake3 dl)

add_test(NAME plugin_packager_test COMMAND plugin_packager_test)

add_executable(rust_ffi_test tests/rust_ffi_test.cpp)
target_include_directories(rust_ffi_test PRIVATE tests)
target_link_libraries(rust_ffi_test PRIVATE harmonics_ffi harmonics BLAKE3::blake3)

add_test(NAME rust_ffi_test COMMAND rust_ffi_test)

add_executable(cli_options_test tests/cli_options_test.cpp)
target_include_directories(cli_options_test PRIVATE tests)
target_link_libraries(cli_options_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME cli_options_test COMMAND cli_options_test)

add_executable(dataset_cache_cli_test tests/dataset_cache_cli_test.cpp)
target_include_directories(dataset_cache_cli_test PRIVATE tests)
target_link_libraries(dataset_cache_cli_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME dataset_cache_cli_test COMMAND dataset_cache_cli_test)

add_executable(dataset_schema_cli_test tests/dataset_schema_cli_test.cpp)
target_include_directories(dataset_schema_cli_test PRIVATE tests)
target_link_libraries(dataset_schema_cli_test PRIVATE harmonics BLAKE3::blake3)

add_test(NAME dataset_schema_cli_test COMMAND dataset_schema_cli_test)
add_executable(graph_info_cli_test tests/graph_info_cli_test.cpp)
target_include_directories(graph_info_cli_test PRIVATE tests)
target_link_libraries(graph_info_cli_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME graph_info_cli_test COMMAND graph_info_cli_test)

add_executable(multi_accelerator_scheduler_test tests/multi_accelerator_scheduler_test.cpp)
target_include_directories(multi_accelerator_scheduler_test PRIVATE tests)
target_link_libraries(multi_accelerator_scheduler_test PRIVATE harmonics BLAKE3::blake3)
add_test(NAME multi_accelerator_scheduler_test COMMAND multi_accelerator_scheduler_test)

# Example programs -------------------------------------------------------------
add_executable(basic_example examples/basic.cpp)
target_link_libraries(basic_example PRIVATE harmonics BLAKE3::blake3)

add_executable(mnist_example examples/mnist.cpp)
target_link_libraries(mnist_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shell_example examples/shell.cpp)
target_link_libraries(shell_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shaders_example examples/shaders.cpp)
target_link_libraries(shaders_example PRIVATE harmonics BLAKE3::blake3)

add_executable(training_shaders_example examples/training_shaders.cpp)
target_link_libraries(training_shaders_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shader_wrapper_cli examples/shader_wrapper_cli.cpp)
target_link_libraries(shader_wrapper_cli PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_example examples/quantum_stub_demo.cpp)
target_link_libraries(quantum_stub_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_large_example examples/quantum_stub_large.cpp)
target_link_libraries(quantum_stub_large_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_complex_example examples/quantum_stub_complex.cpp)
target_link_libraries(quantum_stub_complex_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_backend_example examples/quantum_backend_demo.cpp)
target_link_libraries(quantum_backend_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_backend_large_example examples/quantum_backend_large.cpp)
target_link_libraries(quantum_backend_large_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_hardware_example examples/quantum_hardware_demo.cpp)
target_link_libraries(quantum_hardware_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_hardware_large_example examples/quantum_hardware_large.cpp)
target_link_libraries(quantum_hardware_large_example PRIVATE harmonics BLAKE3::blake3)

# Distributed training with parameter server

# Dataset cache synchronisation demo
add_executable(distributed_dataset_cache_example examples/distributed_dataset_cache_example.cpp)
target_link_libraries(distributed_dataset_cache_example PRIVATE harmonics BLAKE3::blake3)

# Example demonstrating the WebSocket training visualiser
add_executable(training_visualizer_example examples/training_visualizer.cpp)
target_link_libraries(training_visualizer_example PRIVATE harmonics BLAKE3::blake3)

# Simple benchmark comparing CPU, GPU and FPGA execution
add_executable(performance_benchmark examples/performance_benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE harmonics BLAKE3::blake3)

# Extended benchmark suite running multiple tensor sizes
add_executable(benchmark_suite benchmarks/performance_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE harmonics BLAKE3::blake3)

# Convenience target that builds and runs the benchmark suite
add_custom_target(benchmarks
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-benchmarks.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL)

# Sample plugin and loader example
add_library(sample_plugin SHARED examples/plugin_example/sample_plugin.cpp)
target_link_libraries(sample_plugin PRIVATE harmonics BLAKE3::blake3)
set_target_properties(sample_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Plugin demonstrating custom layer implementations
add_library(sample_layer_plugin SHARED examples/layer_plugin/custom_layers.cpp)
target_link_libraries(sample_layer_plugin PRIVATE harmonics BLAKE3::blake3)
set_target_properties(sample_layer_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Quantum hardware stub library used by execute_on_hardware()
add_library(quantum_hw SHARED examples/quantum_hw_stub.cpp)
target_link_libraries(quantum_hw PRIVATE harmonics BLAKE3::blake3)
set_target_properties(quantum_hw PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(plugin_example examples/plugin_example/plugin_example.cpp)
target_link_libraries(plugin_example PRIVATE harmonics BLAKE3::blake3 dl)

# Command line tool -----------------------------------------------------------
add_executable(harmonics_cli src/harmonics.cpp)
target_link_libraries(harmonics_cli PRIVATE harmonics BLAKE3::blake3)

# Graph editing command line tool
add_executable(graph_cli src/graph_cli.cpp)
target_link_libraries(graph_cli PRIVATE harmonics BLAKE3::blake3)
install(TARGETS graph_cli shell_example
        RUNTIME DESTINATION harmonics)

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
