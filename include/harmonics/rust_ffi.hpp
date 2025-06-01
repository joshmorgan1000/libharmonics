#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/dataset.hpp"
#include "harmonics/distributed_scheduler.hpp"
#include "harmonics/flight_io.hpp"
#include "harmonics/grpc_io.hpp"
#include "harmonics/parser.hpp"
#include "harmonics/partition.hpp"
#include "harmonics/runtime.hpp"
#include "harmonics/tcp_io.hpp"
#include "harmonics/training.hpp"
#include "harmonics/wasm_backend.hpp"

namespace harmonics {
extern "C" {

/** Backend identifiers for FFI calls. */
enum harmonics_backend_t {
    HARMONICS_BACKEND_CPU = static_cast<int>(Backend::CPU),
    HARMONICS_BACKEND_GPU = static_cast<int>(Backend::GPU),
    HARMONICS_BACKEND_FPGA = static_cast<int>(Backend::FPGA),
    HARMONICS_BACKEND_WASM = static_cast<int>(Backend::WASM),
    HARMONICS_BACKEND_AUTO = static_cast<int>(Backend::Auto)
};

HarmonicGraph* harmonics_parse_graph(const char* src);
void harmonics_destroy_graph(HarmonicGraph* g);
CycleRuntime* harmonics_create_runtime(const HarmonicGraph* g);
void harmonics_destroy_runtime(CycleRuntime* rt);
void harmonics_forward(CycleRuntime* rt);

/** Simple CSV dataset loader. */
Producer* harmonics_create_csv_producer(const char* path);
/** IDX dataset loader. */
Producer* harmonics_create_idx_producer(const char* path);
/** HDF5 dataset loader. */
Producer* harmonics_create_hdf5_producer(const char* path);
/** Destroy a dataset producer created via the FFI. */
void harmonics_destroy_producer(Producer* p);
/** Destroy a consumer created via the FFI. */
void harmonics_destroy_consumer(Consumer* c);
/** Bind a producer to a graph by name. */
void harmonics_bind_producer(HarmonicGraph* g, const char* name, Producer* p);

/** Partition a graph across multiple backends. */
HarmonicGraph** harmonics_auto_partition(const HarmonicGraph* g,
                                         const harmonics_backend_t* backends, std::size_t count);
void harmonics_destroy_partitions(HarmonicGraph** parts, std::size_t count);

/** Distributed scheduler helpers. */
DistributedScheduler* harmonics_create_distributed_scheduler(HarmonicGraph** parts,
                                                             std::size_t count,
                                                             const harmonics_backend_t* backends,
                                                             bool secure);
void harmonics_destroy_distributed_scheduler(DistributedScheduler* sched);
void harmonics_scheduler_bind_producer(DistributedScheduler* sched, std::size_t part,
                                       const char* name, Producer* p);
CycleRuntime* harmonics_scheduler_runtime(DistributedScheduler* sched, std::size_t part);
void harmonics_scheduler_step(DistributedScheduler* sched);
void harmonics_scheduler_fit(DistributedScheduler* sched, std::size_t epochs);

/** WebAssembly runtime helpers. */
bool harmonics_wasm_available();
bool harmonics_wasm_runtime_available();
bool harmonics_wasm_simd_available();

/** TCP transport helpers. */
Producer* harmonics_create_tcp_producer(const char* host, unsigned short port);
Consumer* harmonics_create_tcp_consumer(const char* host, unsigned short port);
/** gRPC transport helpers. */
Producer* harmonics_create_grpc_producer(const char* host, unsigned short port);
Consumer* harmonics_create_grpc_consumer(const char* host, unsigned short port);
/** Transport helpers disabled. */
#if 0
Producer* harmonics_create_flight_producer(const char* host, unsigned short port);
Consumer* harmonics_create_flight_consumer(const char* host, unsigned short port);
#endif

/** Train the graph for a fixed number of epochs. */
void harmonics_fit(HarmonicGraph* g, std::size_t epochs);
/** Train the graph for a fixed number of seconds. */
void harmonics_fit_for(HarmonicGraph* g, double seconds);

} // extern "C"
} // namespace harmonics
