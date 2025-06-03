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
std::vector<RemoteBinding> cons = {
    {"input", "hostB", 9001, RemoteTransport::TCP, true, 0}}; // compress tensors
RemoteScheduler sched(part, prod, cons);
sched.fit(1);
```

Set `compress` in a binding to enable Zstandard compression for that connection.
Use `max_message_size` with gRPC or Flight transports to control the maximum
allowed message size in bytes.

Multiple remote schedulers can be combined to form a larger distributed graph, allowing flexible placement of partitions across machines.

# Multi-Accelerator Scheduler

The multi-accelerator scheduler automatically partitions a `HarmonicGraph` across CPU, GPU and FPGA backends. Each partition runs in its own `CycleRuntime` so that heterogeneous devices can work together during training or inference.

## Automatic partitioning

Use `auto_partition` with a `DeploymentDescriptor` that lists the desired backends for each partition. Layers are distributed proportionally based on the relative weight of each backend.
The `weight` field of `PartitionOptions` further adjusts this ratio so load can
be balanced more precisely across heterogeneous devices.

```cpp
harmonics::DeploymentDescriptor deploy;
deploy.partitions = {
    {harmonics::Backend::GPU, "", 0, std::nullopt, 1.5}, // GPU with higher weight
    {harmonics::Backend::FPGA},
    {harmonics::Backend::CPU}
};

auto parts = auto_partition(graph, deploy);
harmonics::DistributedScheduler sched(parts, deploy);
```

By default the scheduler assigns weight `4.0` to GPU partitions, `2.0` to FPGA
partitions and `1.0` to CPU partitions. Custom weights override these defaults.

During `step()` or `fit()` the scheduler executes each partition in sequence and forwards boundary tensors through in-memory buses. When `deploy.backend` is set to `Backend::Auto`, the helper `select_accelerator_backend()` chooses the best available device when no explicit backend is specified.

## Deployment options

Each entry in `deploy.partitions` can override the backend and device index for its partition. GPU partitions may set `device_index` while FPGA partitions may use `fpga_device_index` or fall back to the OpenCL platform selection rules.

```cpp
harmonics::DeploymentDescriptor d;
d.partitions.resize(2);
d.partitions[0].backend = harmonics::Backend::GPU;
d.partitions[0].device_index = 1;   // second GPU
d.partitions[0].weight = 2.0;       // heavier load on the GPU

d.partitions[1].backend = harmonics::Backend::FPGA;
d.partitions[1].fpga_device_index = 0; // first OpenCL device
```

The scheduler also accepts the same `secure` flag as other runtimes so chain-of-custody proofs can be enabled across accelerators.

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

# Parameter Server Example

The `distributed_parameter_server_example` program implements a minimal
parameter server using two gRPC streams. One stream distributes the current
parameters while the other collects gradients from a worker process. The server
applies the gradient and broadcasts the updated parameter back to the worker.

## Running the example

Build the project and run the example from the build directory:

```bash
./scripts/run-tests.sh
./build-Release/distributed_parameter_server_example
```

The output prints the parameter value before and after applying the gradient.


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

