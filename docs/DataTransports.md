# Data Transport Layers
# Streaming Input/Output

Harmonics provides several producers and consumers for moving tensors through files, streams and in-memory queues. These helpers implement the `Producer` or `Consumer` interfaces so they can be attached to a `HarmonicGraph` just like dataset loaders.

When a model is partitioned across multiple processes or machines, these producers and consumers form the boundary between layers.  The library does not prescribe a network protocol but makes it easy to ship tensors over any message system implemented by the host application.

## File I/O

`FileProducer` reads tensors from a binary file created by the serialization helpers while `FileConsumer` writes tensors to such a file.

```cpp
#include <harmonics/stream_io.hpp>
#include <harmonics/graph.hpp>

harmonics::FileProducer prod("weights.bin");
// bind `prod` to a graph input
```

## Stream wrappers

`StreamProducer` and `StreamConsumer` operate on existing `std::istream` or `std::ostream` objects. They serialize tensors using the built‑in helpers so any stream type can be used.

```cpp
std::stringstream buf;
harmonics::StreamConsumer writer(buf);
writer.push(tensor);           // serialize to the buffer
harmonics::StreamProducer reader(buf);
auto t = reader.next();        // deserialize from the buffer
```

## Socket transport

On POSIX systems, `SocketProducer` and `SocketConsumer` read and write tensors over a file descriptor using a length prefix. This enables simple interprocess pipelines without additional framing code.

## In‑memory message bus

`MessageBus` implements a thread‑safe queue for tensors. `BusProducer` receives tensors from a bus while `BusConsumer` sends them. This is useful when connecting different parts of a program without sockets or files.

```cpp
auto bus = std::make_shared<harmonics::MessageBus>();
harmonics::BusConsumer cons(bus);
harmonics::BusProducer prod(bus);
cons.push(tensor);
auto r = prod.next();
```

## TCP transport

`TcpProducer` and `TcpConsumer` connect to a host and port using a TCP socket. They share the same length-prefixed framing as `SocketProducer` and `SocketConsumer` so tensors can be streamed across a network.

```cpp
harmonics::TcpConsumer cons("127.0.0.1", 9000);
harmonics::TcpProducer prod("127.0.0.1", 9000);
```

The TCP helpers are only available on POSIX systems.

## gRPC framing

`GrpcProducer` and `GrpcConsumer` operate over a TCP connection using the gRPC
message format. Each tensor is sent with a 1 byte flag followed by a 32‑bit
big‑endian length. Connections perform a lightweight handshake exchanging the
`"HGRP"` magic string so mismatched transports fail fast. The handshake is
carried in the call metadata under the `magic` key.

```cpp
harmonics::GrpcConsumer cons("127.0.0.1", 50051);
harmonics::GrpcProducer prod("127.0.0.1", 50051);
// Optional third argument tunes the maximum message size
harmonics::GrpcConsumer big_cons("127.0.0.1", 50051, 8 * 1024 * 1024);
harmonics::GrpcProducer big_prod("127.0.0.1", 50051, 8 * 1024 * 1024);
// Fourth argument sets a timeout in milliseconds and enables automatic reconnect
harmonics::GrpcConsumer timed_cons("127.0.0.1", 50051, 0, 1000);
harmonics::GrpcProducer timed_prod("127.0.0.1", 50051, 0, 1000);
```

These helpers are also POSIX only.

## WebSocket transport

`WebSocketProducer` and `WebSocketConsumer` provide a simple framing layer over
the WebSocket protocol. They perform a minimal HTTP handshake and then stream
binary tensors using standard WebSocket frames.

```cpp
harmonics::WebSocketConsumer cons("127.0.0.1", 8080, "/tensor");
harmonics::WebSocketProducer prod("127.0.0.1", 8080, "/tensor");
```

These helpers share the same serialization format as the other transports and
are available only on POSIX systems.


## Distributed graph helpers

`StreamingFileProducer` and `StreamingFileConsumer` keep the file handle open so tensors can be streamed incrementally. `SocketServer` and `GrpcServer` listen for incoming connections and return `SocketProducer` or `GrpcProducer` instances for received data, or `SocketConsumer`/`GrpcConsumer` to send data. These utilities make it easier to pipe tensors between graph partitions in separate processes.

## URL helpers

`make_producer()` and `make_consumer()` parse a target string and construct the corresponding transport. This simplifies binding distributed graph boundaries to files or network endpoints.

```cpp
#include <harmonics/transport_helpers.hpp>

auto prod = harmonics::make_producer("tcp:127.0.0.1:9000");
auto cons = harmonics::make_consumer("file:results.bin");
```

Supported schemes are `file`, `socket`, `tcp` and `grpc` (the latter three only on POSIX platforms).

## Apache Arrow Flight

Harmonics now includes lightweight helpers that mimic the core Flight protocol.
`FlightProducer` and `FlightConsumer` reuse the gRPC framing layer but expose
an Arrow‑style API. They can connect to a `TensorFlightServer` running either in
memory or over TCP, making it easy to stream tensors between processes while
reusing optimised Arrow buffers.

```cpp
#include <harmonics/flight_io.hpp>

auto server = std::make_shared<harmonics::TensorFlightServer>();
harmonics::FlightProducer prod(server);
harmonics::FlightConsumer cons(server);

cons.push(tensor);            // send through the server
auto out = prod.next();       // receive the tensor back
```

When Arrow is enabled these helpers can also communicate with external
applications that implement the Flight protocol. This is useful for parameter
servers or remote dataset shards where Arrow binaries are already used.

