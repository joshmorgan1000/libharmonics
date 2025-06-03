# Distributed Scheduler

Harmonics graphs can be partitioned across multiple processes so each segment runs in its own runtime instance. The distributed scheduler coordinates these partitions and forwards tensors across their boundaries.

## Overview

The scheduler expects a set of already partitioned `HarmonicGraph` objects. Each partition defines producers and consumers that mark where tensors enter or leave the local process. When constructing the scheduler these boundaries are connected together using in-memory message buses.

```cpp
std::vector<harmonics::HarmonicGraph> parts = partition_graph(g, 2);
harmonics::DistributedScheduler sched(parts);

// Specify different accelerators for each partition
harmonics::DeploymentDescriptor deploy;
deploy.partitions = {{harmonics::Backend::GPU}, {harmonics::Backend::CPU}};
harmonics::DistributedScheduler accel_sched(parts, deploy);
```

During `step()` or `fit()` the scheduler invokes the `CycleRuntime` for each partition in sequence. Produced tensors are pushed to the appropriate bus so downstream consumers can read them on the next call.

## Remote execution

For cross-process deployments each partition may instead use `RemoteScheduler`. Boundary bindings map producers and consumers to TCP or gRPC transports so tensors move over the network.

```cpp
using harmonics::RemoteBinding;
std::vector<RemoteBinding> prod = {{"output", "hostA", 9000}};
std::vector<RemoteBinding> cons = {
    {"input", "hostB", 9001, RemoteTransport::TCP, true, 0}}; // compress tensors
RemoteScheduler sched(part, prod, cons);
sched.fit(1);
```

Set `compress` in a binding to enable Zstandard compression for that connection.
`max_message_size` can be used with gRPC or Flight transports to tune the
maximum send and receive size in bytes.

Multiple remote schedulers can be combined to form a larger distributed graph, allowing flexible placement of partitions across machines.
