# Dataset Utilities

Harmonics provides a small collection of helpers for loading common dataset formats. These utilities implement the `Producer` interface so they can be bound to graph inputs via `HarmonicGraph::bindProducer`. The loaders remain lightweight but are now production ready, offering solid error handling and streaming options.

## CSV files

`CsvProducer` reads comma separated values from a file or stream and yields one
1‑D float tensor per line. Example:

```cpp
#include <harmonics/dataset.hpp>
#include <harmonics/graph.hpp>

// assuming `g` is a HarmonicGraph with a producer named "input"
auto data = std::make_shared<harmonics::CsvProducer>("train.csv");
g.bindProducer("input", data);
```

`StreamingCsvProducer` provides the same interface but reads the file lazily.
Each call to `next()` parses one line so only a small portion of the dataset is
kept in memory. This is useful when dealing with very large CSV files:

```cpp
auto stream = std::make_shared<harmonics::StreamingCsvProducer>("train.csv");
```

CSV parsing errors now report the row and column of the malformed value to make
debugging easier.

## IDX format

`IdxProducer` loads tensors from the binary IDX format used by datasets like
MNIST. It supports unsigned byte, `float32` and `float64` records of arbitrary
shape.

```cpp
#include <harmonics/dataset.hpp>
#include <harmonics/graph.hpp>

auto images = std::make_shared<harmonics::IdxProducer>("train-images-idx3-ubyte");
g.bindProducer("img", images);
```

Both producers keep the loaded records in memory and cycle through them when
`next()` is called repeatedly.

### Memory-mapped producers

Large files can also be loaded through memory mapping on Unix systems.
`MmapCsvProducer`, `MmapIdxProducer` and `MmapHdf5Producer` map the dataset
into memory and decode each record lazily. This keeps the resident size small
while the operating system pages data in as required.

```cpp
auto csv_mmap = std::make_shared<harmonics::MmapCsvProducer>("train.csv");
auto idx_mmap = std::make_shared<harmonics::MmapIdxProducer>("images.idx");
auto h5_mmap  = std::make_shared<harmonics::MmapHdf5Producer>("data.hdf5");
```

Mapped producers behave like their regular counterparts but start up much
faster for large datasets and allow multiple processes to share cached pages.
Compressed HDF5 containers fall back to eager loading since they cannot be
mapped directly.

## Dataset transformations

Several wrapper producers apply common transformations to another producer.

### ShuffleProducer

`ShuffleProducer` randomises the order of samples returned by an underlying dataset. The permutation is refreshed when all samples have been seen.

```cpp
auto base = std::make_shared<harmonics::CsvProducer>("train.csv");
auto shuffled = std::make_shared<harmonics::ShuffleProducer>(base);
```

Calling `next()` on `shuffled` will yield every row from `train.csv` in a different order each epoch.

### BatchProducer

`BatchProducer` groups consecutive samples into fixed-size batches, emitting a higher-rank tensor. This is typically combined with `ShuffleProducer` to obtain random batches.

```cpp
// shuffle then create batches of 32 samples
auto batched = std::make_shared<harmonics::BatchProducer>(shuffled, 32);
```

Each call to `next()` returns a tensor whose first dimension is the batch size.

### AugmentProducer

`AugmentProducer` applies a user provided function to every sample. The function receives a const reference to the original tensor and must return a transformed tensor.

```cpp
auto augmented = std::make_shared<harmonics::AugmentProducer>(
    base, [](const harmonics::HTensor& t) {
        // example: return the input unchanged
        return t;
    });
```

Any transformation can be implemented inside the lambda, such as image normalisation or random cropping. Scaling is available through the `make_scale` helper which wraps a producer and resizes each tensor by a factor:

```cpp
auto half = harmonics::make_scale(base, 0.5f);
```

This scales each sample with bilinear filtering to preserve quality when
downsizing. The helper is useful for quick normalisation steps without writing
custom augmentation code.

Combining the wrappers yields a typical input pipeline:

```cpp
auto pipeline = std::make_shared<harmonics::BatchProducer>(
    std::make_shared<harmonics::ShuffleProducer>(augmented), 32);
```

### Augmentation pipeline

`AugmentationPipeline` chains multiple augmentation steps together. Each step is
represented by a callback receiving a tensor and returning a transformed copy.
The helpers in `augmentation.hpp` provide common operations like random crops or
noise injection. When constructed with caching enabled the pipeline stores the
processed tensors during the first pass and reuses them on subsequent epochs.

```cpp
std::vector<harmonics::AugmentationPipeline::Fn> steps;
steps.push_back(harmonics::flip_horizontal);
steps.push_back(harmonics::flip_vertical);
steps.push_back([](const harmonics::HTensor& t) {
    return harmonics::rotate(t, 180.f);
});
steps.push_back([](const harmonics::HTensor& t) {
    return harmonics::scale(t, 0.5f);
});
steps.push_back([](const harmonics::HTensor& t) {
    return harmonics::add_noise(t, 0.1f);
});
auto pipe = harmonics::make_augmentation_pipeline(base, steps, true);
```

## Asynchronous loading

`AsyncProducer` wraps another producer and prefetches samples on a background
thread. This is useful when the underlying dataset incurs I/O latency. The
wrapper maintains a small queue whose capacity defaults to one sample.

```cpp
auto async = std::make_shared<harmonics::AsyncProducer>(dataset);
```

## Other formats

`TFRecordProducer` reads records from TensorFlow TFRecord files and yields a byte
tensor for each record.

`CocoJsonProducer` extracts bounding boxes from COCO annotation files, returning a
tensor with four floats per bounding box.

`HttpProducer` streams tensor records over HTTP. When constructed with a cache
file path it stores the downloaded records on disk and reloads them on subsequent
runs:

```cpp
harmonics::HttpProducer prod("127.0.0.1", 8080, "/data", true, "cache.bin");
```

`AsyncHttpProducer` performs the same HTTP request using non-blocking sockets
and feeds downloaded records to a background queue. Samples become available
as soon as they are received, hiding network latency from the main thread:

```cpp
auto async_http =
    std::make_shared<harmonics::AsyncHttpProducer>("127.0.0.1", 8080, "/data");
```

`HttpHdf5Producer` downloads a complete HDF5 container over HTTP and exposes
its records just like `Hdf5Producer`. When given a cache file path the
downloaded bytes are stored locally and reused on subsequent runs:

```cpp
harmonics::HttpHdf5Producer loader("127.0.0.1", 8080, "/dataset.hdf5", true,
                                  "dataset.hdf5");
```

If the cache file exists the producer skips the network request and reads the
records directly from disk.

`Hdf5Producer` and `Hdf5Consumer` implement a minimal container format inspired
by HDF5. The consumer serialises tensors into a file while the producer reads
them back later. This makes it easy to snapshot preprocessed datasets:

The container header now stores a 64‑bit record count so very large datasets are
supported without truncation.

```cpp
// save two samples
harmonics::Hdf5Consumer writer("train.hdf5");
writer.push(sample_a);
writer.push(sample_b);

// reload them
harmonics::Hdf5Producer loader("train.hdf5");
auto a = loader.next();
auto b = loader.next();
```

### Checkpointable Hdf5Consumer

Long running training jobs may need to periodically flush results to disk without
losing previously written samples. The checkpointable variant of
`Hdf5Consumer` can reopen an existing container and append additional tensors.
It updates the record count in the file header so the data can be read back with
`Hdf5Producer` just like a freshly created file.

```cpp
harmonics::CheckpointableHdf5Consumer writer("train.hdf5");
writer.push(new_sample);
```

## Dataset caching

`CachedProducer` wraps another producer and writes every sample to a small HDF5
container. When the cache file already exists it is loaded directly instead of
reading from the underlying dataset. This avoids repeated parsing or downloads
across runs.

```cpp
auto base = std::make_shared<harmonics::CsvProducer>("train.csv");
auto cached = std::make_shared<harmonics::CachedProducer>(
    base, "train_cache.hdf5", true, true);
```

The first execution stores all tensors in `train_cache.hdf5`. Subsequent runs
can pass `nullptr` as the inner producer to reuse the cached records:

```cpp
harmonics::CachedProducer from_cache(nullptr, "train_cache.hdf5");
```

Compression and schema validation are controlled by the constructor arguments.

## Dataset schema validation

Dataset loaders can optionally check that each record matches an expected shape and data type. Pass a `DatasetSchema` when constructing a producer or call `validate_schema()` on an existing one to enable the checks. The loader throws `std::runtime_error` if any sample deviates from the schema.

```cpp
harmonics::DatasetSchema spec;
spec.dtype = harmonics::HTensor::DType::Float32;
spec.shape = {1, 784};

auto csv = std::make_shared<harmonics::CsvProducer>("train.csv");
harmonics::validate_schema(*csv, spec);
```

Schema validation is disabled by default so existing examples work unchanged.

You can also validate an entire HDF5 dataset from the command line using
`dataset_schema_cli`:

```bash
dataset_schema_cli images.hdf5 f32 784
```

## Error handling

Dataset loaders validate their inputs and throw descriptive errors when
encountering malformed files or invalid values. CSV readers detect
non-numeric fields, truncated rows and unterminated quotes, raising
`std::runtime_error` in such cases. Similar checks exist for other
formats to ensure corrupted data does not silently propagate through the
pipeline.
Error messages include the row and column number where possible to make
debugging data issues straightforward.


## Dataset conversion CLI

The `dataset_convert` tool converts simple CSV or IDX datasets into the minimal HDF5 container format used by Harmonics. This allows datasets to be preprocessed once and loaded efficiently during training.

```bash
dataset_convert (--csv|--idx) <in> [-o out.hdf5]
```

Use `--csv` for comma separated values or `--idx` for the binary IDX format commonly used by MNIST. The optional `-o/--out` argument specifies the output file name and defaults to `out.hdf5`.

Example:

```bash
dataset_convert --idx train-images-idx3-ubyte -o images.hdf5
dataset_convert --idx train-labels-idx1-ubyte -o labels.hdf5
```

The resulting files can be loaded with `Hdf5Producer` like any other dataset source.

## Distributed dataset cache

`DistributedCachedProducer` synchronises cache files between machines when datasets are shared across nodes. When the local cache is missing it is automatically downloaded from a remote producer. On shutdown the cache can be uploaded to a remote consumer so other workers reuse the processed records.

A helper CLI, `dataset_cache_cli`, transfers cache files over TCP:

```
dataset_cache_cli download <path> <host> <port>
dataset_cache_cli upload <path> <host> <port>
dataset_cache_cli serve-download <path> [port]
dataset_cache_cli serve-upload <path> [port]
dataset_cache_cli hash <path>
```

`serve-*` starts a small server that sends or receives a cache file. The `hash` command prints the BLAKE3 digest of a cache for integrity checks. Downloads automatically resume if the destination file already contains a partial cache so interrupted transfers continue where they left off.

### Distributed cache example

The `distributed_dataset_cache_example` program demonstrates two nodes exchanging a cached dataset over a socket server. Build and run the example from the build directory:

```bash
./scripts/run-tests.sh
./build-Release/distributed_dataset_cache_example
```

The output lists the tensors received by the node that downloads the cache.

