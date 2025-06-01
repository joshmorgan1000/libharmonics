# Runtime and Scheduling
# Distributed Scheduler

Harmonics graphs can be partitioned across multiple processes so each segment runs in its own runtime instance. The distributed scheduler coordinates these partitions and forwards tensors across their boundaries.

## Overview

The scheduler expects a set of already partitioned `HarmonicGraph` objects. Each partition defines producers and consumers that mark where tensors enter or leave the local process. When constructing the scheduler these boundaries are connected together using in-memory message buses.

```cpp
std::vector<harmonics::HarmonicGraph> parts = partition_graph(g, 2);
harmonics::DistributedScheduler sched(parts);
```

During `step()` or `fit()` the scheduler invokes the `CycleRuntime` for each partition in sequence. Produced tensors are pushed to the appropriate bus so downstream consumers can read them on the next call.

## Remote execution

For cross-process deployments each partition may instead use `RemoteScheduler`. Boundary bindings map producers and consumers to TCP or gRPC transports so tensors move over the network.

```cpp
using harmonics::RemoteBinding;
std::vector<RemoteBinding> prod = {{"output", "hostA", 9000}};
std::vector<RemoteBinding> cons = {{"input", "hostB", 9001}};
RemoteScheduler sched(part, prod, cons);
sched.fit(1);
```

Multiple remote schedulers can be combined to form a larger distributed graph, allowing flexible placement of partitions across machines.

# Multi-Accelerator Scheduler

The multi-accelerator scheduler automatically partitions a `HarmonicGraph` across CPU, GPU and FPGA backends. Each partition runs in its own `CycleRuntime` so that heterogeneous devices can work together during training or inference.

## Automatic partitioning

Use `auto_partition` with a `DeploymentDescriptor` that lists the desired backends for each partition. Layers are distributed proportionally based on the relative weight of each backend.

```cpp
harmonics::DeploymentDescriptor deploy;
deploy.partitions = {
    {harmonics::Backend::GPU},
    {harmonics::Backend::FPGA},
    {harmonics::Backend::CPU}
};

auto parts = auto_partition(graph, deploy);
harmonics::DistributedScheduler sched(parts, deploy);
```

During `step()` or `fit()` the scheduler executes each partition in sequence and forwards boundary tensors through in-memory buses. When `deploy.backend` is set to `Backend::Auto`, the helper `select_accelerator_backend()` chooses the best available device when no explicit backend is specified.

## Deployment options

Each entry in `deploy.partitions` can override the backend and device index for its partition. GPU partitions may set `device_index` while FPGA partitions follow the OpenCL platform selection rules.

```cpp
harmonics::DeploymentDescriptor d;
d.partitions.resize(2);
d.partitions[0].backend = harmonics::Backend::GPU;
d.partitions[0].device_index = 1;   // second GPU

d.partitions[1].backend = harmonics::Backend::CPU;
```

The scheduler also accepts the same `secure` flag as other runtimes so chain-of-custody proofs can be enabled across accelerators.

# Distributed Parameter Server Example

The `distributed_parameter_server.cpp` example illustrates how a basic SGD-style parameter server can be built with the provided gRPC transport. A small tensor is sent from the server to workers, updated with gradients and streamed back.

## Overview

The program defines a simple `ParameterServer` class that holds a single float parameter. Gradients are received over gRPC from a producer, an SGD update is applied and the new parameter is published via a server so workers can pull the latest value.

## Running the example

Build the project with the helper script and execute the example binary from the build directory:

```bash
./scripts/run-tests.sh
./build-Release/distributed_parameter_server_example
```

The program prints the initial value of the parameter, sends a dummy gradient and shows the updated value after the server applies the update.

# Distributed Scheduler Example

The `distributed_scheduler_example` program demonstrates how partitions of a graph can communicate over different network transports using `RemoteScheduler`.
It runs a tiny graph split into two parts and forwards tensors across the boundary using TCP sockets and gRPC.

## Running the example

Build the project and run the example from the build directory:

```bash
./scripts/run-tests.sh
./build-Release/distributed_scheduler_example
```

The output shows the dimension of the tensor received by the second partition for each transport.


# Runtime Checkpointing

Long running training jobs can fail or need to be paused. Harmonics allows the complete state of a `CycleRuntime` to be saved to disk and later restored so work can continue without starting from scratch.

## Saving the runtime state

Use `save_checkpoint()` to serialise the current weights, optimiser parameters and precision policy to a file.

```cpp
harmonics::CycleRuntime rt(graph);
// ... training loop ...
rt.save_checkpoint("checkpoint.bin");
```

## Resuming from a checkpoint

Recreate the runtime with the same graph and call `load_checkpoint()` before resuming training. All internal counters and parameters are restored from the file.

```cpp
harmonics::CycleRuntime rt(graph);
rt.load_checkpoint("checkpoint.bin");
// continue training
```

Checkpoints are portable across machines as long as the underlying compute backend is available.

