option(ENABLE_NETWORK_TESTS "Build tests requiring network access" OFF)
option(ENABLE_PLUGIN_TESTS "Build tests requiring plugin support" OFF)
option(ENABLE_CLI_TESTS "Build tests for command line tools" OFF)

add_executable(dummy_test dummy_test.cpp)
target_link_libraries(dummy_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME dummy_test COMMAND dummy_test)

add_executable(function_registry_test function_registry_test.cpp)
target_link_libraries(function_registry_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME function_registry_test COMMAND function_registry_test)

add_executable(function_registry_thread_test function_registry_thread_test.cpp)
target_link_libraries(function_registry_thread_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME function_registry_thread_test COMMAND function_registry_thread_test)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(parser_test parser_test.cpp)
target_link_libraries(parser_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME parser_test COMMAND parser_test)

add_executable(graph_test graph_test.cpp)
target_link_libraries(graph_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME graph_test COMMAND graph_test)

add_executable(cycle_runtime_test cycle_runtime_test.cpp)
target_link_libraries(cycle_runtime_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME cycle_runtime_test COMMAND cycle_runtime_test)

add_executable(constant_slab_integration_test constant_slab_integration_test.cpp)
target_link_libraries(constant_slab_integration_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME constant_slab_integration_test COMMAND constant_slab_integration_test)

add_executable(threaded_runtime_test threaded_runtime_test.cpp)
target_link_libraries(threaded_runtime_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME threaded_runtime_test COMMAND threaded_runtime_test)

add_executable(training_tap_test training_tap_test.cpp)
target_link_libraries(training_tap_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME training_tap_test COMMAND training_tap_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(runtime_api_test runtime_api_test.cpp)
    target_link_libraries(runtime_api_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME runtime_api_test COMMAND runtime_api_test)
endif()

add_executable(mnist_example_test mnist_example_test.cpp)
target_link_libraries(mnist_example_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME mnist_example_test COMMAND mnist_example_test)

add_executable(mnist_training_cycle_test mnist_training_cycle_test.cpp)
target_link_libraries(mnist_training_cycle_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME mnist_training_cycle_test COMMAND mnist_training_cycle_test)

add_executable(secure_mode_test secure_mode_test.cpp)
target_include_directories(secure_mode_test PRIVATE
    ${protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS})
target_link_libraries(secure_mode_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME secure_mode_test COMMAND secure_mode_test)

add_executable(gpu_backend_test gpu_backend_test.cpp)
target_link_libraries(gpu_backend_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME gpu_backend_test COMMAND gpu_backend_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(vulkan_device_test vulkan_device_test.cpp)
    target_link_libraries(vulkan_device_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME vulkan_device_test COMMAND vulkan_device_test)

    add_executable(cuda_device_test cuda_device_test.cpp)
    target_link_libraries(cuda_device_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME cuda_device_test COMMAND cuda_device_test)
endif()

if(ENABLE_HARDWARE_TESTS)
    add_executable(opencl_device_test opencl_device_test.cpp)
    target_link_libraries(opencl_device_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME opencl_device_test COMMAND opencl_device_test)

    add_executable(quantum_device_test quantum_device_test.cpp)
    target_link_libraries(quantum_device_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME quantum_device_test COMMAND quantum_device_test)
endif()

if(Vulkan_FOUND)
    add_executable(vulkan_backend_test vulkan_backend_test.cpp)
    target_link_libraries(vulkan_backend_test PRIVATE harmonics micro_gtest BLAKE3::blake3 Vulkan::Vulkan)
    add_test(NAME vulkan_backend_test COMMAND vulkan_backend_test)
endif()

if(ENABLE_HARDWARE_TESTS)
    add_executable(vulkan_backend_enabled_test vulkan_backend_enabled_test.cpp)
    target_link_libraries(vulkan_backend_enabled_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME vulkan_backend_enabled_test COMMAND vulkan_backend_enabled_test)
endif()

add_executable(cuda_backend_enabled_test cuda_backend_enabled_test.cpp)
target_link_libraries(cuda_backend_enabled_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME cuda_backend_enabled_test COMMAND cuda_backend_enabled_test)

add_executable(gpu_fallback_test gpu_fallback_test.cpp)
target_link_libraries(gpu_fallback_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
add_test(NAME gpu_fallback_test COMMAND gpu_fallback_test)

add_executable(gpu_shader_wrapper_test gpu_shader_wrapper_test.cpp)
target_link_libraries(gpu_shader_wrapper_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME gpu_shader_wrapper_test COMMAND gpu_shader_wrapper_test)

add_executable(int8_matmul_test int8_matmul_test.cpp)
target_link_libraries(int8_matmul_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME int8_matmul_test COMMAND int8_matmul_test)

add_executable(int8_determinism_test int8_determinism_test.cpp)
target_link_libraries(int8_determinism_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME int8_determinism_test COMMAND int8_determinism_test)

add_executable(cross_target_determinism_test cross_target_determinism_test.cpp)
target_link_libraries(cross_target_determinism_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME cross_target_determinism_test COMMAND cross_target_determinism_test)

add_executable(int8_cross_target_determinism_test int8_cross_target_determinism_test.cpp)
target_link_libraries(int8_cross_target_determinism_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME int8_cross_target_determinism_test COMMAND int8_cross_target_determinism_test)

add_executable(int_sgd_update_test int_sgd_update_test.cpp)
target_link_libraries(int_sgd_update_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME int_sgd_update_test COMMAND int_sgd_update_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(int4_matmul_test int4_matmul_test.cpp)
    target_link_libraries(int4_matmul_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME int4_matmul_test COMMAND int4_matmul_test)

    add_executable(int3_matmul_test int3_matmul_test.cpp)
    target_link_libraries(int3_matmul_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME int3_matmul_test COMMAND int3_matmul_test)
endif()

add_executable(gpu_kernel_api_test gpu_kernel_api_test.cpp)
target_link_libraries(gpu_kernel_api_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME gpu_kernel_api_test COMMAND gpu_kernel_api_test)

add_executable(wrapper_ring_buffer_test wrapper_ring_buffer_test.cpp)
target_link_libraries(wrapper_ring_buffer_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME wrapper_ring_buffer_test COMMAND wrapper_ring_buffer_test)

add_executable(accelerator_selection_cpu_test accelerator_selection_cpu_test.cpp)
target_link_libraries(accelerator_selection_cpu_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME accelerator_selection_cpu_test COMMAND accelerator_selection_cpu_test)

add_executable(accelerator_selection_gpu_test accelerator_selection_gpu_test.cpp)
target_link_libraries(accelerator_selection_gpu_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
add_test(NAME accelerator_selection_gpu_test COMMAND accelerator_selection_gpu_test)

add_executable(fpga_fallback_test fpga_fallback_test.cpp)
target_link_libraries(fpga_fallback_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
add_test(NAME fpga_fallback_test COMMAND fpga_fallback_test)

add_executable(gpu_fpga_fallback_integration_test gpu_fpga_fallback_integration_test.cpp)
target_link_libraries(gpu_fpga_fallback_integration_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
add_test(NAME gpu_fpga_fallback_integration_test COMMAND gpu_fpga_fallback_integration_test)

add_executable(fpga_backend_test fpga_backend_test.cpp)
target_link_libraries(fpga_backend_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME fpga_backend_test COMMAND fpga_backend_test)

add_executable(wasm_backend_test wasm_backend_test.cpp)
target_link_libraries(wasm_backend_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME wasm_backend_test COMMAND wasm_backend_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(accelerator_selection_wasm_test accelerator_selection_wasm_test.cpp)
    target_link_libraries(accelerator_selection_wasm_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME accelerator_selection_wasm_test COMMAND accelerator_selection_wasm_test)
endif()

add_executable(wasm_api_test wasm_api_test.cpp)
target_link_libraries(wasm_api_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME wasm_api_test COMMAND wasm_api_test)

add_executable(fpga_kernel_compiler_test fpga_kernel_compiler_test.cpp)
target_link_libraries(fpga_kernel_compiler_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME fpga_kernel_compiler_test COMMAND fpga_kernel_compiler_test)

add_executable(precision_utils_test precision_utils_test.cpp)
target_link_libraries(precision_utils_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME precision_utils_test COMMAND precision_utils_test)

add_executable(layer_functions_test layer_functions_test.cpp)
target_link_libraries(layer_functions_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME layer_functions_test COMMAND layer_functions_test)

add_executable(activation_library_test activation_library_test.cpp)
target_link_libraries(activation_library_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME activation_library_test COMMAND activation_library_test)

if(ENABLE_HARDWARE_TESTS)
    add_executable(dataset_producer_test dataset_producer_test.cpp)
    target_link_libraries(dataset_producer_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME dataset_producer_test COMMAND dataset_producer_test)

    add_executable(augmentation_test augmentation_test.cpp)
    target_link_libraries(augmentation_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME augmentation_test COMMAND augmentation_test)
endif()

add_executable(async_producer_test async_producer_test.cpp)
target_link_libraries(async_producer_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME async_producer_test COMMAND async_producer_test)

if(Protobuf_FOUND)
    add_executable(model_import_test model_import_test.cpp)
    target_link_libraries(model_import_test PRIVATE harmonics micro_gtest BLAKE3::blake3 onnx_tiny)
    add_test(NAME model_import_test COMMAND model_import_test)

    add_executable(small_llm_test small_llm_test.cpp)
    target_link_libraries(small_llm_test PRIVATE harmonics micro_gtest BLAKE3::blake3 onnx_tiny)
    add_test(NAME small_llm_test COMMAND small_llm_test)

    add_executable(onnx_export_test onnx_export_test.cpp)
    target_link_libraries(onnx_export_test PRIVATE harmonics micro_gtest BLAKE3::blake3 onnx_tiny)
    add_test(NAME onnx_export_test COMMAND onnx_export_test)
endif()

add_executable(producer_consumer_io_test producer_consumer_io_test.cpp)
target_link_libraries(producer_consumer_io_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME producer_consumer_io_test COMMAND producer_consumer_io_test)

add_executable(transport_helpers_test transport_helpers_test.cpp)
target_link_libraries(transport_helpers_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME transport_helpers_test COMMAND transport_helpers_test)

if(ENABLE_NETWORK_TESTS)
    add_executable(tcp_io_test tcp_io_test.cpp)
    target_link_libraries(tcp_io_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME tcp_io_test COMMAND tcp_io_test)

    add_executable(websocket_io_test websocket_io_test.cpp)
    target_link_libraries(websocket_io_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME websocket_io_test COMMAND websocket_io_test)

    add_executable(grpc_io_test grpc_io_test.cpp)
    target_link_libraries(grpc_io_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME grpc_io_test COMMAND grpc_io_test)

    add_executable(spark_dataset_test spark_dataset_test.cpp)
    target_link_libraries(spark_dataset_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME spark_dataset_test COMMAND spark_dataset_test)

    add_executable(distributed_io_test distributed_io_test.cpp)
    target_link_libraries(distributed_io_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME distributed_io_test COMMAND distributed_io_test)
endif()

add_executable(serialization_test serialization_test.cpp)
target_link_libraries(serialization_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME serialization_test COMMAND serialization_test)

add_executable(introspection_test introspection_test.cpp)
target_link_libraries(introspection_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME introspection_test COMMAND introspection_test)

add_executable(runtime_memory_test runtime_memory_test.cpp)
target_link_libraries(runtime_memory_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME runtime_memory_test COMMAND runtime_memory_test)

add_executable(memory_transfer_stats_test memory_transfer_stats_test.cpp)
target_link_libraries(memory_transfer_stats_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME memory_transfer_stats_test COMMAND memory_transfer_stats_test)

add_executable(graph_edit_test graph_edit_test.cpp)
target_link_libraries(graph_edit_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME graph_edit_test COMMAND graph_edit_test)

add_executable(graph_diff_test graph_diff_test.cpp)
target_link_libraries(graph_diff_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME graph_diff_test COMMAND graph_diff_test)

add_executable(kernel_cache_test kernel_cache_test.cpp)
target_link_libraries(kernel_cache_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME kernel_cache_test COMMAND kernel_cache_test)

add_executable(shader_cache_test shader_cache_test.cpp)
target_link_libraries(shader_cache_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME shader_cache_test COMMAND shader_cache_test)

add_executable(cuda_kernel_cache_test cuda_kernel_cache_test.cpp)
target_link_libraries(cuda_kernel_cache_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME cuda_kernel_cache_test COMMAND cuda_kernel_cache_test)

add_executable(partition_test partition_test.cpp)
target_link_libraries(partition_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME partition_test COMMAND partition_test)

add_executable(auto_partition_test auto_partition_test.cpp)
target_link_libraries(auto_partition_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME auto_partition_test COMMAND auto_partition_test)

add_executable(quantum_stub_test quantum_stub_test.cpp)
target_link_libraries(quantum_stub_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME quantum_stub_test COMMAND quantum_stub_test)

add_executable(quantum_backend_test quantum_backend_test.cpp)
target_link_libraries(quantum_backend_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME quantum_backend_test COMMAND quantum_backend_test)

add_executable(quantum_hardware_test quantum_hardware_test.cpp)
target_link_libraries(quantum_hardware_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME quantum_hardware_test COMMAND quantum_hardware_test)


# Plugin tests require dlopen support which may not be available
# in the minimal CI environment. Disable them by default.
if(ENABLE_PLUGIN_TESTS)
    add_library(test_plugin SHARED test_plugin.cpp)
    target_link_libraries(test_plugin PRIVATE harmonics micro_gtest BLAKE3::blake3)
    set_target_properties(test_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(plugin_test plugin_test.cpp)
    target_link_libraries(plugin_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
    add_test(NAME plugin_test COMMAND plugin_test)

    add_executable(plugin_async_test plugin_async_test.cpp)
    target_link_libraries(plugin_async_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
    add_test(NAME plugin_async_test COMMAND plugin_async_test)

    add_executable(plugin_reload_test plugin_reload_test.cpp)
    target_link_libraries(plugin_reload_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
    add_test(NAME plugin_reload_test COMMAND plugin_reload_test)

    add_executable(plugin_packager_test plugin_packager_test.cpp)
    target_link_libraries(plugin_packager_test PRIVATE harmonics micro_gtest BLAKE3::blake3 dl)
    add_test(NAME plugin_packager_test COMMAND plugin_packager_test)
endif()

add_executable(rust_ffi_test rust_ffi_test.cpp)
target_link_libraries(rust_ffi_test PRIVATE harmonics_ffi harmonics micro_gtest BLAKE3::blake3)
add_test(NAME rust_ffi_test COMMAND rust_ffi_test)

if(ENABLE_CLI_TESTS)
    add_executable(cli_options_test cli_options_test.cpp)
    target_link_libraries(cli_options_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME cli_options_test COMMAND cli_options_test)

    add_executable(dataset_cache_cli_test dataset_cache_cli_test.cpp)
    target_link_libraries(dataset_cache_cli_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME dataset_cache_cli_test COMMAND dataset_cache_cli_test)

    add_executable(dataset_schema_cli_test dataset_schema_cli_test.cpp)
    target_link_libraries(dataset_schema_cli_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME dataset_schema_cli_test COMMAND dataset_schema_cli_test)

    add_executable(graph_info_cli_test graph_info_cli_test.cpp)
    target_link_libraries(graph_info_cli_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME graph_info_cli_test COMMAND graph_info_cli_test)

    add_executable(multi_accelerator_scheduler_test multi_accelerator_scheduler_test.cpp)
    target_link_libraries(multi_accelerator_scheduler_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
    add_test(NAME multi_accelerator_scheduler_test COMMAND multi_accelerator_scheduler_test)
endif()

add_executable(fp_lr_schedule_test fp_lr_schedule_test.cpp)
target_link_libraries(fp_lr_schedule_test PRIVATE harmonics micro_gtest BLAKE3::blake3)
add_test(NAME fp_lr_schedule_test COMMAND fp_lr_schedule_test)
