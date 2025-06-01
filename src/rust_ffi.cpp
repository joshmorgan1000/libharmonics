#include "harmonics/rust_ffi.hpp"
#include <chrono>
#include <vector>

namespace harmonics {
extern "C" {

HarmonicGraph* harmonics_parse_graph(const char* src) {
    Parser parser{src};
    auto ast = parser.parse_declarations();
    return new HarmonicGraph(build_graph(ast));
}

void harmonics_destroy_graph(HarmonicGraph* g) { delete g; }

CycleRuntime* harmonics_create_runtime(const HarmonicGraph* g) { return new CycleRuntime{*g}; }

void harmonics_destroy_runtime(CycleRuntime* rt) { delete rt; }

void harmonics_forward(CycleRuntime* rt) { rt->forward(); }

Producer* harmonics_create_csv_producer(const char* path) {
    return new CsvProducer{std::string(path)};
}

Producer* harmonics_create_idx_producer(const char* path) {
    return new IdxProducer{std::string(path)};
}

Producer* harmonics_create_hdf5_producer(const char* path) {
    return new Hdf5Producer{std::string(path)};
}

void harmonics_destroy_producer(Producer* p) { delete p; }

void harmonics_destroy_consumer(Consumer* c) { delete c; }

Producer* harmonics_create_tcp_producer(const char* host, unsigned short port) {
    return new TcpProducer{std::string(host), port};
}

Consumer* harmonics_create_tcp_consumer(const char* host, unsigned short port) {
    return new TcpConsumer{std::string(host), port};
}

Producer* harmonics_create_grpc_producer(const char* host, unsigned short port) {
    return new GrpcProducer{std::string(host), port};
}

Consumer* harmonics_create_grpc_consumer(const char* host, unsigned short port) {
    return new GrpcConsumer{std::string(host), port};
}

/* Transport helpers disabled
Producer* harmonics_create_flight_producer(const char* host, unsigned short port) {
    return new FlightProducer{std::string(host), port};
}

Consumer* harmonics_create_flight_consumer(const char* host, unsigned short port) {
    return new FlightConsumer{std::string(host), port};
}
*/

void harmonics_bind_producer(HarmonicGraph* g, const char* name, Producer* p) {
    if (!g || !p || !name)
        return;
    g->bindProducer(name, std::shared_ptr<Producer>(p));
}

HarmonicGraph** harmonics_auto_partition(const HarmonicGraph* g,
                                         const harmonics_backend_t* backends, std::size_t count) {
    if (!g || !backends || count == 0)
        return nullptr;
    DeploymentDescriptor d;
    d.partitions.resize(count);
    for (std::size_t i = 0; i < count; ++i)
        d.partitions[i].backend = static_cast<Backend>(backends[i]);
    auto parts = auto_partition(*g, d);
    HarmonicGraph** out = new HarmonicGraph*[parts.size()];
    for (std::size_t i = 0; i < parts.size(); ++i)
        out[i] = new HarmonicGraph(std::move(parts[i]));
    return out;
}

void harmonics_destroy_partitions(HarmonicGraph** parts, std::size_t count) {
    if (!parts)
        return;
    for (std::size_t i = 0; i < count; ++i)
        delete parts[i];
    delete[] parts;
}

DistributedScheduler* harmonics_create_distributed_scheduler(HarmonicGraph** parts,
                                                             std::size_t count,
                                                             const harmonics_backend_t* backends,
                                                             bool secure) {
    if (!parts || !backends || count == 0)
        return nullptr;
    std::vector<HarmonicGraph> vec;
    vec.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        vec.push_back(std::move(*parts[i]));
        delete parts[i];
    }
    delete[] parts;
    DeploymentDescriptor d;
    d.secure = secure;
    d.partitions.resize(count);
    for (std::size_t i = 0; i < count; ++i)
        d.partitions[i].backend = static_cast<Backend>(backends[i]);
    return new DistributedScheduler(std::move(vec), d);
}

void harmonics_destroy_distributed_scheduler(DistributedScheduler* sched) { delete sched; }

void harmonics_scheduler_bind_producer(DistributedScheduler* sched, std::size_t part,
                                       const char* name, Producer* p) {
    if (!sched || !p || !name)
        return;
    sched->bindProducer(part, name, std::shared_ptr<Producer>(p));
}

CycleRuntime* harmonics_scheduler_runtime(DistributedScheduler* sched, std::size_t part) {
    if (!sched)
        return nullptr;
    return &sched->runtime(part);
}

void harmonics_scheduler_step(DistributedScheduler* sched) {
    if (sched)
        sched->step();
}

void harmonics_scheduler_fit(DistributedScheduler* sched, std::size_t epochs) {
    if (sched)
        sched->fit(epochs);
}

bool harmonics_wasm_available() { return wasm_available(); }
bool harmonics_wasm_runtime_available() { return wasm_runtime_available(); }
bool harmonics_wasm_simd_available() { return wasm_simd_available(); }

void harmonics_fit(HarmonicGraph* g, std::size_t epochs) {
    if (!g)
        return;
    g->fit(epochs, nullptr);
}

void harmonics_fit_for(HarmonicGraph* g, double seconds) {
    if (!g)
        return;
    auto dur = std::chrono::duration<double>(seconds);
    g->fit(dur, nullptr);
}

} // extern "C"
} // namespace harmonics
