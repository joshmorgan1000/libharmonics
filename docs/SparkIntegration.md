# Spark Integration

This design document outlines how Harmonics graphs can be executed within an Apache Spark cluster using the distributed scheduler. It covers partitioning strategies, data transport and deployment considerations.

## Overview

Harmonics graphs can be partitioned into multiple `HarmonicGraph` objects which are then executed in sequence by the `DistributedScheduler`. Spark can orchestrate these partitions across its worker nodes by packaging each partition into a Spark task. Communication between tasks occurs via standard Harmonics transports (TCP, gRPC or message buses).

## Partitioning strategies

1. **Layer based partitioning** – large graphs are split at layer boundaries so each Spark task processes a subset of the layers. This keeps memory usage low per executor but requires tensors to be streamed across tasks.
2. **Data parallelism** – the same graph is replicated across many tasks with each partition processing different input records. Aggregation occurs via a reducer stage that merges gradients or outputs.
3. **Hybrid approach** – combine layer splits with data parallel replicas for complex models. Spark's scheduler can launch multiple copies of each partition to scale horizontally.

## Data transport mechanisms

Spark tasks communicate using Harmonics transports. For local clusters message buses can be shared through the JVM using JNI bindings. In distributed deployments a TCP or gRPC transport is recommended. Each task binds its producers and consumers to the appropriate transport endpoints using `RemoteBinding` objects. Compression via Zstandard can be enabled to reduce network traffic when tensors are large.

## Configuration

When running on Spark, each executor launches the Harmonics runtime inside its task. Important configuration options include:

- **Backend selection** – choose CPU, CUDA or FPGA backends via `DeploymentDescriptor` so each executor uses available hardware.
- **Partition assignments** – precompute graph partitions and distribute them to tasks using Spark's broadcast mechanism.
- **Transport ports** – allocate unique TCP ports for each boundary pair or use gRPC load balancing. Ports can be derived from the Spark partition id to avoid clashes.
- **Secure mode** – enable `secure` in the deployment descriptor when zero‑knowledge proofs are required. Proof data is forwarded through the same transports.

## Performance considerations

- Use Spark's built‑in partitioning to colocate tasks that exchange large tensors on the same worker when possible. This allows the use of shared memory transports which are faster than network sockets.
- For data parallel training aggregate gradients using a tree reduction to minimize shuffle overhead.
- Cache frequently used datasets in HDFS or a distributed cache so each executor can perform local reads rather than remote fetches.

## Example workflow

1. Pre-partition a graph into N segments using `partition_graph`.
2. Broadcast the partitions to executors as serialized blobs.
3. Launch a Spark job where each task constructs a `DistributedScheduler` with its assigned partition.
4. Bind producers and consumers using TCP or gRPC endpoints derived from the Spark partition id.
5. Call `fit()` or `step()` inside the task to process data.
6. Collect results or proof strings using standard Spark actions.

This approach allows complex Harmonics models to scale across a Spark cluster while retaining the library's flexible execution semantics.

## Spark deployment

With the dataset helpers now available, Spark executors can instantiate
`Producer` objects directly inside each task. The helpers mirror the
single‑node API so existing code works unchanged:

```cpp
auto csv = std::make_shared<harmonics::CsvProducer>(dataset_path);
auto batched = std::make_shared<harmonics::BatchProducer>(csv, batch_size);
scheduler.bindProducer("input", batched);
```

Each executor can maintain its own cache and augmentation pipeline, or
share datasets through HDFS and other distributed stores. Broadcasting the
dataset paths keeps the tasks lightweight while still leveraging the full
`Producer` feature set. See [Datasets.md](Datasets.md) for the available
helpers.

