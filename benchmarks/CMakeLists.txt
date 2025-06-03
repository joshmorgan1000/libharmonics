# Extended benchmark suite running multiple tensor sizes
add_executable(benchmark_suite performance_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE harmonics BLAKE3::blake3)

# Convenience target that builds and runs the benchmark suite
add_custom_target(benchmarks
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/run-benchmarks.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    USES_TERMINAL)