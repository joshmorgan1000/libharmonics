# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------

add_executable(multi_gpu_training multi_gpu_training.cpp)
target_include_directories(multi_gpu_training PRIVATE include)
target_link_libraries(multi_gpu_training PRIVATE harmonics BLAKE3::blake3)

add_executable(basic_example basic.cpp)
target_link_libraries(basic_example PRIVATE harmonics BLAKE3::blake3)

add_executable(mnist_example mnist.cpp)
target_link_libraries(mnist_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shell_example shell.cpp)
target_link_libraries(shell_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shaders_example shaders.cpp)
target_link_libraries(shaders_example PRIVATE harmonics BLAKE3::blake3)

add_executable(training_shaders_example training_shaders.cpp)
target_link_libraries(training_shaders_example PRIVATE harmonics BLAKE3::blake3)

add_executable(shader_wrapper_cli shader_wrapper_cli.cpp)
target_link_libraries(shader_wrapper_cli PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_example quantum_stub_demo.cpp)
target_link_libraries(quantum_stub_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_large_example quantum_stub_large.cpp)
target_link_libraries(quantum_stub_large_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_stub_complex_example quantum_stub_complex.cpp)
target_link_libraries(quantum_stub_complex_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_backend_example quantum_backend_demo.cpp)
target_link_libraries(quantum_backend_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_backend_large_example quantum_backend_large.cpp)
target_link_libraries(quantum_backend_large_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_hardware_example quantum_hardware_demo.cpp)
target_link_libraries(quantum_hardware_example PRIVATE harmonics BLAKE3::blake3)

add_executable(quantum_hardware_large_example quantum_hardware_large.cpp)
target_link_libraries(quantum_hardware_large_example PRIVATE harmonics BLAKE3::blake3)

# Distributed training with parameter server
add_executable(distributed_parameter_server_example distributed_parameter_server_example.cpp)
target_link_libraries(distributed_parameter_server_example PRIVATE harmonics BLAKE3::blake3)

# Cluster server and client demonstrating distributed execution
add_executable(cluster_server cluster_server.cpp)
target_link_libraries(cluster_server PRIVATE harmonics BLAKE3::blake3)

add_executable(cluster_client cluster_client.cpp)
target_link_libraries(cluster_client PRIVATE harmonics BLAKE3::blake3)

# Dataset cache synchronisation demo
add_executable(distributed_dataset_cache_example distributed_dataset_cache_example.cpp)
target_link_libraries(distributed_dataset_cache_example PRIVATE harmonics BLAKE3::blake3)

# Example demonstrating the WebSocket training visualiser
add_executable(training_visualizer_example training_visualizer.cpp)
target_link_libraries(training_visualizer_example PRIVATE harmonics BLAKE3::blake3)

# Simple benchmark comparing CPU, GPU and FPGA execution
add_executable(performance_benchmark performance_benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE harmonics BLAKE3::blake3)

# Sample plugin and loader example
add_library(sample_plugin SHARED plugin_example/sample_plugin.cpp)
target_link_libraries(sample_plugin PRIVATE harmonics BLAKE3::blake3)
set_target_properties(sample_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Plugin demonstrating custom layer implementations
add_library(sample_layer_plugin SHARED layer_plugin/custom_layers.cpp)
target_link_libraries(sample_layer_plugin PRIVATE harmonics BLAKE3::blake3)
set_target_properties(sample_layer_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Quantum hardware backend used by execute_on_hardware()
add_library(quantum_hw SHARED quantum_hw_backend.cpp)
target_link_libraries(quantum_hw PRIVATE harmonics BLAKE3::blake3)
set_target_properties(quantum_hw PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(plugin_example plugin_example/plugin_example.cpp)
target_link_libraries(plugin_example PRIVATE harmonics BLAKE3::blake3 dl)
