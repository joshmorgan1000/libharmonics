# Cluster Deployment

This guide demonstrates how to run a Harmonics graph across two machines using the `RemoteScheduler`.
Two small programs are provided: `cluster_server` hosts the second partition and
relays tensors over TCP, while `cluster_client` runs the first partition and
forwards its boundary output to the server.

## Building the examples

Compile the project and examples:

```bash
./scripts/run-tests.sh
```

The binaries `cluster_server` and `cluster_client` are placed in the build
directory.

## Running on a cluster

1. Copy `cluster_server` to the machine that will host the second partition and
   run it. The program listens on ports **6000** and **6001**.

   ```bash
   ./build-Release/cluster_server
   ```

2. On another machine, run `cluster_client` and pass the hostname or IP address
   of the server machine:

   ```bash
   ./build-Release/cluster_client server_host
   ```

The client sends a tensor to the server which runs the remaining layers of the
graph. The server prints the dimension of the tensor received from the client.

This setup mirrors a minimal cluster deployment where each partition executes in
its own process and communicates via the distributed scheduler.
