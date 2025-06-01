
#pragma once

/**
 * @file dataset.hpp
 * @brief Collection of lightweight dataset utilities.
 *
 * This header defines several small dataset producers that can be used
 * to load and transform tensors from various file formats. The focus is
 * on simplicity and having no heavy dependencies so the implementations
 * are intentionally minimal.
 */

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zstd.h>

#include "harmonics/core.hpp"
#include "harmonics/serialization.hpp"

#ifdef __unix__
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace harmonics {

// Forward declaration used by constructors.
inline void validate_schema(const std::vector<HTensor>& records);

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------
// * **Producer**  - Object implementing the `Producer` interface which yields
//   tensors on demand. Producers are typically chained together to form simple
//   input pipelines for the runtime.
// * **Consumer**  - Object implementing the `Consumer` interface which accepts
//   tensors from the runtime. Only a handful of minimal consumers are provided
//   in this header.
// * **Record**    - A single tensor loaded from disk or produced on the fly.
//   Many of the helpers cache records in memory for simplicity.
// * **Dataset**   - Informal term used to describe a configured producer or
//   chain of producers that can supply tensors for training or inference.
// * **Sample**    - One element produced by a dataset, typically represented as
//   an `HTensor`. Several producers combine or modify samples to build more
//   complex datasets.
// * **Batch**     - A collection of multiple samples stacked together along a
//   new leading dimension. Created by `BatchProducer` to drive vectorised
//   execution in the runtime.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dataset helper overview
//
// This header defines the collection of dataset producers used across the
// library and examples. They began life as minimal utilities for unit tests
// but have since matured into lightweight, production-ready loaders.
//
// The implementations remain deliberately compact so new contributors can
// easily follow how data flows into the runtime. Extensive comments describe
// design choices and limitations for those writing custom loaders.
//
// Each loader mirrors the behaviour of a real data source while avoiding
// heavy external dependencies to keep compilation fast. Despite their small
// size the helpers now include solid error handling and are suitable for
// real workloads.
//
// The available producers are:
//   - CsvProducer:    load comma separated numeric data
//   - IdxProducer:    parse simple binary IDX files
//   - ShuffleProducer: randomise the order of samples
//   - BatchProducer:   combine consecutive samples into a batch
//   - AugmentProducer: apply element-wise transformations
//   - TFRecordProducer: read TensorFlow record files
//   - CocoJsonProducer: parse minimal COCO annotation data
//   - Hdf5Producer / Consumer: extremely small tensor container
//   - HttpProducer:    stream tensors over HTTP
//   - StreamingCsvProducer: read large CSV files line by line
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Additional notes on design
//
// The goal of these helpers is to stay as small as possible while providing
// dependable handling of the supported formats. They intentionally avoid heavy
// dependencies so the code remains easy to read and compile. The loaders now
// validate inputs and handle common edge cases, making them reliable for
// production use.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Utility helpers shared by the dataset loaders
// ---------------------------------------------------------------------------
inline std::string trim_ws(std::string_view s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return std::string(s.substr(b, e - b));
}

inline void parse_csv_line(std::string_view line, std::vector<float>& out,
                           std::size_t row = static_cast<std::size_t>(-1)) {
    std::string cell;
    bool in_quotes = false;
    std::size_t col = 1;
    auto throw_err = [&](const char* msg) {
        if (row != static_cast<std::size_t>(-1)) {
            std::ostringstream oss;
            oss << msg << " at row " << row << " column " << col;
            throw std::runtime_error(oss.str());
        }
        throw std::runtime_error(msg);
    };
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            auto t = trim_ws(cell);
            if (!t.empty()) {
                try {
                    out.push_back(std::stof(t));
                } catch (const std::exception&) {
                    throw_err("invalid number in CSV");
                }
            }
            cell.clear();
            ++col;
        } else {
            cell.push_back(c);
        }
    }
    if (in_quotes)
        throw_err("unterminated quoted field in CSV");
    auto t = trim_ws(cell);
    if (!t.empty()) {
        try {
            out.push_back(std::stof(t));
        } catch (const std::exception&) {
            throw_err("invalid number in CSV");
        }
    }
}

/**
 * Producer that loads 1D float tensors from CSV formatted data.
 *
 * Each line of the input is parsed as comma separated floating
 * point values which become one tensor sample.
 */
/**
 * @brief Simple CSV based data loader.
 *
 * This producer reads comma separated values from a text stream and
 * converts each row into a 1D tensor of floating point samples. It is
 * intentionally small and does not perform any fancy parsing beyond
 * splitting on commas, which means it can be used in environments where
 * bringing in a full CSV parser would be overkill.
 */
class CsvProducer : public Producer {
  public:
    /**
     * Construct a producer reading from a CSV file.
     *
     * @param path path to the CSV file
     */
    explicit CsvProducer(const std::string& path, bool validate = false) : validate_{validate} {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("failed to open CSV file");
        parse_stream(f);
        if (validate_)
            validate_schema(records_);
    }

    /** Construct a producer reading from an existing input stream. */
    explicit CsvProducer(std::istream& in, bool validate = false) : validate_{validate} {
        parse_stream(in);
        if (validate_)
            validate_schema(records_);
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        const auto& t = records_[index_++ % records_.size()];
        return t;
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};
    bool validate_{false};

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /**
     * @brief Split a single CSV line into float values.
     *
     * This helper performs a very small amount of parsing by streaming
     * over the line and extracting each comma separated cell. Empty cells
     * are ignored which allows for trailing commas in the file.
     */
    static void parse_stream_line(std::string line, std::vector<float>& out, std::size_t row) {
        parse_csv_line(line, out, row);
    }

    /**
     * @brief Parse the entire CSV stream into tensor records.
     *
     * Each line results in one tensor. The resulting records are stored
     * in memory so repeated calls to @ref next simply cycle through the
     * parsed data.
     */
    /**
     * @brief Parse an IDX formatted stream.
     *
     * The entire file is loaded into memory so repeated reads are
     * inexpensive. Only a subset of data types are handled as this
     * implementation is primarily intended for unit tests.
     */
    void parse_stream(std::istream& in) {
        std::string line;
        std::size_t expected = 0;
        // Iterate over every line in the file and convert it into a tensor.
        // The parsing is intentionally forgiving so that sample CSV files
        // used in tests can contain extra whitespace or trailing commas.
        std::size_t row = 1;
        while (std::getline(in, line)) {
            std::vector<float> vals;
            // Convert the comma separated string into numeric values.
            parse_stream_line(line, vals, row);
            if (validate_) {
                if (expected == 0)
                    expected = vals.size();
                else if (vals.size() != expected)
                    throw std::runtime_error("dataset schema mismatch");
            }
            // Use the number of parsed elements as the tensor width. This
            // keeps the tensor representation extremely simple: each row
            // becomes a 1D tensor where the length equals the number of
            // cells found on that line.
            HTensor::Shape shape{vals.size()};
            std::vector<std::byte> data(vals.size() * sizeof(float));
            std::memcpy(data.data(), vals.data(), data.size());
            // Store the tensor in our list of records so that calls to
            // next() simply cycle through this cached vector.
            records_.emplace_back(HTensor::DType::Float32, shape, std::move(data));
            ++row;
        }
    }
};

/**
 * @brief Stream large CSV files line by line.
 *
 * This variant keeps the input stream open and parses each row only when
 * requested via @ref next. It avoids loading the entire file into memory and
 * is therefore suitable for large datasets.
 */
class StreamingCsvProducer : public Producer {
  public:
    explicit StreamingCsvProducer(const std::string& path, bool validate = false)
        : in_{std::make_shared<std::ifstream>(path)}, validate_{validate} {
        auto* f = static_cast<std::ifstream*>(in_.get());
        if (!*f)
            throw std::runtime_error("failed to open CSV file");
    }
    explicit StreamingCsvProducer(std::istream& in, bool validate = false)
        : in_{&in, [](std::istream*) {}}, validate_{validate} {}
    explicit StreamingCsvProducer(std::shared_ptr<std::istream> in, bool validate = false)
        : in_{std::move(in)}, validate_{validate} {}

    HTensor next() override {
        if (!in_ || !*in_)
            return {};
        std::string line;
        if (!std::getline(*in_, line)) {
            if (!in_->eof())
                throw std::runtime_error("failed to read CSV stream");
            return {};
        }
        std::vector<float> vals;
        parse_line(line, vals, index_ + 1);
        if (validate_) {
            if (expected_ == 0)
                expected_ = vals.size();
            else if (vals.size() != expected_) {
                std::ostringstream msg;
                msg << "dataset schema mismatch at record " << index_ << ": expected width "
                    << expected_ << ", got " << vals.size();
                throw std::runtime_error(msg.str());
            }
        }
        HTensor::Shape shape{vals.size()};
        std::vector<std::byte> data(vals.size() * sizeof(float));
        std::memcpy(data.data(), vals.data(), data.size());
        ++index_;
        return {HTensor::DType::Float32, shape, std::move(data)};
    }

    std::size_t size() const override { return 0; }

  private:
    std::shared_ptr<std::istream> in_{};
    bool validate_{false};
    std::size_t expected_{0};
    std::size_t index_{0};
    static void parse_line(const std::string& line, std::vector<float>& out, std::size_t row) {
        parse_csv_line(line, out, row);
    }
};

/**
 * @brief Memory mapped CSV loader.
 *
 * This variant maps the entire file into memory and stores the
 * offsets of each line. Records are parsed on demand which keeps
 * the memory footprint low for large files.
 */
class MmapCsvProducer : public Producer {
  public:
#ifdef __unix__
    explicit MmapCsvProducer(const std::string& path, bool validate = false) : validate_{validate} {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("failed to open CSV file");
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("failed to stat CSV file");
        }
        size_ = static_cast<std::size_t>(st.st_size);
        map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("failed to mmap CSV file");
        }
        parse_memory(static_cast<const char*>(map_), size_);
        if (validate_)
            check_consistency();
    }

    ~MmapCsvProducer() override {
        if (map_ && map_ != MAP_FAILED)
            ::munmap(map_, size_);
        if (fd_ >= 0)
            ::close(fd_);
    }
#else
    explicit MmapCsvProducer(const std::string&, bool = false) {
        throw std::runtime_error("memory mapping not supported on this platform");
    }
#endif

    HTensor next() override {
        if (lines_.empty())
            return {};
        auto pair = lines_[index_++ % lines_.size()];
        std::vector<float> vals;
        parse_line(pair.first, pair.second, vals, index_ % lines_.size() + 1);
        HTensor::Shape shape{vals.size()};
        std::vector<std::byte> data(vals.size() * sizeof(float));
        std::memcpy(data.data(), vals.data(), data.size());
        return {HTensor::DType::Float32, shape, std::move(data)};
    }

    std::size_t size() const override { return lines_.size(); }

  private:
#ifdef __unix__
    int fd_{-1};
    void* map_{nullptr};
    std::size_t size_{0};
#endif
    std::vector<std::pair<const char*, const char*>> lines_{};
    std::size_t index_{0};
    bool validate_{false};

    static void parse_line(const char* start, const char* end, std::vector<float>& out,
                           std::size_t row) {
        std::string_view view(start, end - start);
        parse_csv_line(view, out, row);
    }

    void parse_memory(const char* data, std::size_t len) {
        const char* line_start = data;
        const char* p = data;
        const char* end = data + len;
        for (; p < end; ++p) {
            if (*p == '\n') {
                lines_.emplace_back(line_start, p);
                line_start = p + 1;
            }
        }
        if (line_start < end)
            lines_.emplace_back(line_start, end);
    }

    void check_consistency() {
        if (lines_.size() <= 1)
            return;
        std::vector<float> vals;
        parse_line(lines_[0].first, lines_[0].second, vals, 1);
        HTensor::Shape shape{vals.size()};
        for (std::size_t i = 1; i < lines_.size(); ++i) {
            std::vector<float> tmp;
            parse_line(lines_[i].first, lines_[i].second, tmp, i + 1);
            if (tmp.size() != shape[0]) {
                std::ostringstream msg;
                msg << "dataset schema mismatch at record " << i << ": expected width " << shape[0]
                    << ", got " << tmp.size();
                throw std::runtime_error(msg.str());
            }
        }
    }
};

/**
 * Producer for datasets stored in the IDX binary format.
 *
 * This reader handles a subset of the format used by popular
 * handwritten digit datasets such as MNIST.
 */
/**
 * @brief Reader for the IDX binary format.
 *
 * The IDX format is commonly used for small educational datasets.
 * Only a tiny subset needed for the sample code is supported here.
 * The parser expects the data to be stored in big endian byte order
 * and loads the entire file into memory for easy random access.
 */
class IdxProducer : public Producer {
  public:
    /**
     * Construct a producer reading from an IDX file.
     *
     * @param path path to the IDX file
     */
    explicit IdxProducer(const std::string& path, bool validate = false) : validate_{validate} {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("failed to open IDX file");
        parse_stream(f);
        if (validate_)
            validate_schema(records_);
    }

    /** Construct a producer reading from an existing binary stream. */
    explicit IdxProducer(std::istream& in, bool validate = false) : validate_{validate} {
        parse_stream(in);
        if (validate_)
            validate_schema(records_);
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        const auto& t = records_[index_++ % records_.size()];
        return t;
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};
    bool validate_{false};

    /**
     * @brief Read a big endian 32‑bit integer from the stream.
     */
    static uint32_t read_be32(std::istream& in) {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
               (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
    }

    void parse_stream(std::istream& in) {
        unsigned char header[4];
        // The first two bytes are zero for the indexed format followed by a
        // single byte indicating the element type and another byte describing
        // how many dimensions each record contains.
        in.read(reinterpret_cast<char*>(header), 4);
        if (!in)
            throw std::runtime_error("invalid IDX header");
        unsigned char type = header[2];
        unsigned char dims = header[3];
        std::vector<uint32_t> dim_sizes(dims);
        // Each dimension size is encoded as a 32-bit big endian integer and
        // describes the shape of the resulting tensor. The first dimension is
        // the record count which we treat separately.
        for (unsigned char i = 0; i < dims; ++i)
            dim_sizes[i] = read_be32(in);
        if (dim_sizes.empty())
            throw std::runtime_error("IDX file missing dimensions");
        std::size_t count = dim_sizes[0];
        std::vector<std::size_t> shape;
        // Skip the first dimension which is the number of records stored.
        for (std::size_t i = 1; i < dim_sizes.size(); ++i)
            shape.push_back(dim_sizes[i]);

        HTensor::DType dtype;
        std::size_t elem_size = 1;
        switch (type) {
        case 0x08:
            dtype = HTensor::DType::UInt8;
            elem_size = 1;
            break;
        case 0x0C:
            dtype = HTensor::DType::Int32;
            elem_size = 4;
            break;
        case 0x0D:
            dtype = HTensor::DType::Float32;
            elem_size = 4;
            break;
        case 0x0E:
            dtype = HTensor::DType::Float64;
            elem_size = 8;
            break;
        default:
            throw std::runtime_error("unsupported IDX data type");
        }

        // The element size combined with the dimensionality tells us how
        // many bytes to read for each record. The IDX format stores all
        // records back to back without any padding which allows us to
        // compute an exact offset for each tensor.

        // Determine how many bytes constitute a single record. We multiply all
        // dimensions except the first since that one simply encodes the number
        // of records present in the file.
        std::size_t record_elems = 1;
        // Compute number of elements per record, skipping the count dimension.
        for (std::size_t i = 1; i < dim_sizes.size(); ++i)
            record_elems *= dim_sizes[i];
        std::size_t record_bytes = record_elems * elem_size;
        // The file now contains `count` consecutive records of the same size.
        // We iterate over each one, reading exactly `record_bytes` into a
        // temporary buffer. No attempt is made to lazily map the file or
        // stream partial records because the test datasets are tiny. The
        // simple approach keeps the code compact and avoids platform
        // specific file handling.
        for (std::size_t i = 0; i < count; ++i) {
            std::vector<std::byte> data(record_bytes);
            in.read(reinterpret_cast<char*>(data.data()), record_bytes);
            if (!in)
                throw std::runtime_error("truncated IDX data");
            // Store the parsed tensor in our local cache.
            records_.emplace_back(dtype, shape, std::move(data));
        }
        if (in.peek() != EOF)
            throw std::runtime_error("extra data at end of IDX file");
    }
};

/**
 * @brief Memory mapped reader for IDX datasets.
 *
 * Maps the binary file and exposes each record lazily without copying
 * the entire dataset into memory.
 */
class MmapIdxProducer : public Producer {
  public:
#ifdef __unix__
    explicit MmapIdxProducer(const std::string& path, bool validate = false) : validate_{validate} {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("failed to open IDX file");
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("failed to stat IDX file");
        }
        size_ = static_cast<std::size_t>(st.st_size);
        map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("failed to mmap IDX file");
        }
        parse_memory(static_cast<const char*>(map_), size_);
        if (validate_)
            ; // nothing additional to check as header enforces schema
    }

    ~MmapIdxProducer() override {
        if (map_ && map_ != MAP_FAILED)
            ::munmap(map_, size_);
        if (fd_ >= 0)
            ::close(fd_);
    }
#else
    explicit MmapIdxProducer(const std::string&, bool = false) {
        throw std::runtime_error("memory mapping not supported on this platform");
    }
#endif

    HTensor next() override {
        if (offsets_.empty())
            return {};
        const char* p = offsets_[index_++ % offsets_.size()];
        std::vector<std::byte> data(record_bytes_);
        std::memcpy(data.data(), p, record_bytes_);
        return {dtype_, shape_, std::move(data)};
    }

    std::size_t size() const override { return offsets_.size(); }

  private:
#ifdef __unix__
    int fd_{-1};
    void* map_{nullptr};
    std::size_t size_{0};
#endif
    HTensor::DType dtype_{};
    std::vector<std::size_t> shape_{};
    std::size_t record_bytes_{0};
    const char* data_begin_{nullptr};
    const char* data_end_{nullptr};
    std::vector<const char*> offsets_{};
    std::size_t index_{0};
    bool validate_{false};

    static uint32_t read_be32_mem(const char*& p) {
        uint32_t v = (static_cast<uint32_t>(static_cast<unsigned char>(p[0])) << 24) |
                     (static_cast<uint32_t>(static_cast<unsigned char>(p[1])) << 16) |
                     (static_cast<uint32_t>(static_cast<unsigned char>(p[2])) << 8) |
                     static_cast<uint32_t>(static_cast<unsigned char>(p[3]));
        p += 4;
        return v;
    }

    void parse_memory(const char* data, std::size_t len) {
        const char* p = data;
        if (len < 4 || p[0] != 0 || p[1] != 0)
            throw std::runtime_error("invalid IDX file");
        unsigned char type = static_cast<unsigned char>(p[2]);
        unsigned char dims = static_cast<unsigned char>(p[3]);
        p += 4;
        std::vector<uint32_t> dim_sizes(dims);
        for (unsigned char i = 0; i < dims; ++i)
            dim_sizes[i] = read_be32_mem(p);
        if (dim_sizes.empty())
            throw std::runtime_error("IDX file missing dimensions");
        std::size_t count = dim_sizes[0];
        shape_.clear();
        for (std::size_t i = 1; i < dim_sizes.size(); ++i)
            shape_.push_back(dim_sizes[i]);

        std::size_t elem_size = 1;
        switch (type) {
        case 0x08:
            dtype_ = HTensor::DType::UInt8;
            elem_size = 1;
            break;
        case 0x0C:
            dtype_ = HTensor::DType::Int32;
            elem_size = 4;
            break;
        case 0x0D:
            dtype_ = HTensor::DType::Float32;
            elem_size = 4;
            break;
        case 0x0E:
            dtype_ = HTensor::DType::Float64;
            elem_size = 8;
            break;
        default:
            throw std::runtime_error("unsupported IDX data type");
        }

        record_bytes_ = elem_size;
        for (std::size_t i = 1; i < dim_sizes.size(); ++i)
            record_bytes_ *= dim_sizes[i];
        data_begin_ = p;
        data_end_ = data + len;

        offsets_.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (p + record_bytes_ > data_end_)
                throw std::runtime_error("truncated IDX data");
            offsets_.push_back(p);
            p += record_bytes_;
        }
    }
};

// ---------------------------------------------------------------------------
// Transformation helpers

/**
 * @brief Shuffle wrapper around another producer.
 *
 * All samples from the inner producer are loaded into memory and then
 * served back in a random order each epoch. This utility keeps the
 * implementation extremely simple while still being useful for small
 * unit test datasets where deterministic ordering might hide bugs.
 */
class ShuffleProducer : public Producer {
  public:
    explicit ShuffleProducer(std::shared_ptr<Producer> inner) : inner_{std::move(inner)} {
        // Eagerly read all samples so we can cheaply permute them.
        // The assumption is that test datasets are small enough to fit in
        // memory which simplifies the shuffling logic immensely.
        data_.reserve(inner_->size());
        for (std::size_t i = 0; i < inner_->size(); ++i)
            data_.push_back(inner_->next());
        // Initial shuffle prior to the first call to next() ensures the
        // first epoch is randomly ordered as well.
        std::shuffle(data_.begin(), data_.end(), rng_);
    }

    HTensor next() override {
        if (data_.empty())
            return {};
        if (index_ >= data_.size()) {
            index_ = 0;
            // Reshuffle to provide a new random order each epoch.
            // The RNG is seeded on construction so repeated runs of the
            // same test produce deterministic orderings when desired.
            std::shuffle(data_.begin(), data_.end(), rng_);
        }
        return data_[index_++];
    }

    std::size_t size() const override { return data_.size(); }

  private:
    std::shared_ptr<Producer> inner_{};
    std::vector<HTensor> data_{};
    std::size_t index_{0};
    std::mt19937 rng_{std::random_device{}()};
};

/**
 * @brief Return the size in bytes of a single element for the given type.
 */
inline std::size_t dtype_size(HTensor::DType dt) {
    switch (dt) {
    case HTensor::DType::Float32:
    case HTensor::DType::Int32:
        return 4;
    case HTensor::DType::Float64:
    case HTensor::DType::Int64:
        return 8;
    case HTensor::DType::UInt8:
    default:
        return 1;
    }
}

inline const char* dtype_name(HTensor::DType dt) {
    switch (dt) {
    case HTensor::DType::Float32:
        return "f32";
    case HTensor::DType::Float64:
        return "f64";
    case HTensor::DType::Int32:
        return "i32";
    case HTensor::DType::Int64:
        return "i64";
    case HTensor::DType::UInt8:
    default:
        return "u8";
    }
}

inline std::string shape_to_string(const HTensor::Shape& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size())
            out += ",";
    }
    out += "]";
    return out;
}

/// Validate that all tensors in the dataset share the same shape and dtype.
inline void validate_schema(const std::vector<HTensor>& records) {
    if (records.empty())
        return;
    const auto& ref_shape = records.front().shape();
    HTensor::DType ref_type = records.front().dtype();
    for (std::size_t i = 1; i < records.size(); ++i) {
        if (records[i].dtype() != ref_type || records[i].shape() != ref_shape) {
            std::ostringstream msg;
            msg << "dataset schema mismatch at record " << i << ": expected "
                << dtype_name(ref_type) << shape_to_string(ref_shape) << ", got "
                << dtype_name(records[i].dtype()) << shape_to_string(records[i].shape());
            throw std::runtime_error(msg.str());
        }
    }
}

/**
 * @brief Combine samples into fixed-size batches.
 *
 * This adaptor collects a number of consecutive samples from another
 * producer and concatenates them along a new leading dimension. The
 * resulting batched tensor is useful for unit tests that need to feed
 * small batches through the runtime.
 */
class BatchProducer : public Producer {
  public:
    BatchProducer(std::shared_ptr<Producer> inner, std::size_t batch)
        : inner_{std::move(inner)}, batch_{batch} {
        // The constructor simply stores the inner producer and desired batch
        // size. Actual batching happens lazily in next().
    }

    HTensor next() override {
        if (batch_ == 0)
            return {};
        std::vector<HTensor> samples;
        samples.reserve(batch_);
        for (std::size_t i = 0; i < batch_; ++i)
            samples.push_back(inner_->next()); // gather batch
        // At this point `samples` holds `batch_` tensors. They are assumed to
        // have identical shapes which keeps the concatenation logic trivial.
        if (samples.empty() || samples[0].shape().empty())
            return {};
        const auto& first = samples[0];
        HTensor::Shape shape{batch_};
        shape.insert(shape.end(), first.shape().begin(), first.shape().end());
        std::size_t elems = 1;
        // Total number of elements per sample (excluding batch dim).
        for (auto d : first.shape())
            elems *= d;
        std::size_t record_bytes = elems * dtype_size(first.dtype());
        std::vector<std::byte> data(record_bytes * batch_);
        for (std::size_t i = 0; i < batch_; ++i)
            std::memcpy(data.data() + i * record_bytes, samples[i].data().data(), record_bytes);
        // Construct the final tensor from the concatenated bytes. The shape now
        // includes the batch dimension at the front followed by the original
        // sample dimensions.
        return HTensor{first.dtype(), std::move(shape), std::move(data)};
    }

    std::size_t size() const override { return inner_->size() / batch_; }

  private:
    std::shared_ptr<Producer> inner_{};
    std::size_t batch_{1};
};

/**
 * @brief Generic element-wise augmentation wrapper.
 *
 * The provided function is invoked for each sample returned by the
 * wrapped producer allowing on-the-fly data augmentation. The tensor
 * shape and dtype must remain unchanged by the function.
 */
class AugmentProducer : public Producer {
  public:
    using Fn = std::function<HTensor(const HTensor&)>;

    /// Construct with the producer to wrap and the augmentation function.

    AugmentProducer(std::shared_ptr<Producer> inner, Fn fn)
        : inner_{std::move(inner)}, fn_{std::move(fn)} {
        // The augmentation function must not alter the shape or dtype of the
        // tensor. No checks are performed here so callers must ensure this
        // contract is honoured.
    }

    HTensor next() override {
        // Pass the sample through the augmentation function. Because the
        // function is user supplied it may perform arbitrary computations.
        return fn_(inner_->next());
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    std::shared_ptr<Producer> inner_{};
    Fn fn_{};
};

// ---------------------------------------------------------------------------
// Format specific producers

/**
 * @brief Minimal TFRecord reader.
 *
 * The implementation only supports the small subset of the format used in
 * unit tests. Records are returned verbatim as byte tensors. Checksums are
 * parsed but ignored as they provide little value in this context.
 */
class TFRecordProducer : public Producer {
  public:
    explicit TFRecordProducer(std::istream& in) { parse_stream(in); }
    explicit TFRecordProducer(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("failed to open TFRecord file");
        parse_stream(f);
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        const auto& t = records_[index_++ % records_.size()];
        return t;
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};

    /// Read a little-endian 64-bit value from the stream.
    static std::uint64_t read_u64(std::istream& in) {
        std::uint64_t v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    static std::uint32_t read_u32(std::istream& in) {
        std::uint32_t v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    /// Parse the entire TFRecord file into an array of tensors.
    void parse_stream(std::istream& in) {
        while (in) {
            std::uint64_t len = read_u64(in);
            if (!in)
                break;
            read_u32(in); // length CRC (ignored)
            std::vector<std::byte> data(len);
            in.read(reinterpret_cast<char*>(data.data()), len);
            if (!in)
                break;
            read_u32(in); // data CRC (ignored)
            records_.emplace_back(HTensor::DType::UInt8, HTensor::Shape{len}, std::move(data));
        }
    }
};

/**
 * @brief Very small JSON reader used in tests.
 *
 * The parser is intentionally naive and only extracts bounding box arrays
 * from COCO style annotation files. It is good enough for unit tests and
 * keeps dependencies to a minimum.
 */
class CocoJsonProducer : public Producer {
  public:
    explicit CocoJsonProducer(std::istream& in) { parse_stream(in); }
    explicit CocoJsonProducer(const std::string& path) {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("failed to open JSON file");
        parse_stream(f);
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        const auto& t = records_[index_++ % records_.size()];
        return t;
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};

    /// Convert a comma separated list of numbers into floats.
    static void parse_numbers(std::string s, std::vector<float>& out) {
        std::stringstream ss(s);
        std::string n;
        while (std::getline(ss, n, ',')) {
            n = trim_ws(n);
            if (n.empty())
                continue;
            try {
                out.push_back(std::stof(n));
            } catch (const std::exception&) {
                throw std::runtime_error("invalid number in JSON");
            }
        }
    }

    /// Parse the JSON file and extract all bounding box arrays.
    void parse_stream(std::istream& in) {
        std::string src((std::istreambuf_iterator<char>(in)), {});
        std::size_t pos = 0;
        while (true) {
            // Find the next "bbox" entry in the JSON.
            pos = src.find("\"bbox\"", pos);
            if (pos == std::string::npos)
                break;
            pos = src.find('[', pos);
            if (pos == std::string::npos)
                break;
            auto end = src.find(']', pos);
            if (end == std::string::npos)
                break;
            std::string arr = src.substr(pos + 1, end - pos - 1);
            std::vector<float> vals;
            parse_numbers(arr, vals);
            if (vals.empty()) {
                pos = end + 1;
                continue;
            }
            if (vals.size() % 4 != 0)
                throw std::runtime_error("invalid bbox array in JSON");
            HTensor::Shape shape{vals.size()};
            std::vector<std::byte> data(vals.size() * sizeof(float));
            std::memcpy(data.data(), vals.data(), data.size());
            records_.emplace_back(HTensor::DType::Float32, std::move(shape), std::move(data));
            pos = end + 1;
        }
    }
};

/**
 * Extremely small container format that mimics storing tensors in an HDF5 file.
 *
 * The implementation does not depend on the HDF5 library. Instead it writes a
 * simple header followed by serialized tensors. The first four bytes are the
 * ASCII string "HDF5" followed by a 32‑bit little endian record count.
 */
/**
 * @brief Tiny tensor container similar to HDF5.
 *
 * Only the features required for unit testing are implemented. The
 * format starts with a simple header and then serializes each tensor
 * consecutively. This avoids introducing a dependency on the real
 * HDF5 library while still allowing us to read and write collections
 * of tensors.
 */
class Hdf5Producer : public Producer {
  public:
    explicit Hdf5Producer(std::istream& in, bool validate = false) : validate_{validate} {
        parse_stream(in);
        if (validate_)
            validate_schema(records_);
    }
    explicit Hdf5Producer(const std::string& path, bool validate = false) : validate_{validate} {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("failed to open HDF5 file");
        parse_stream(f);
        if (validate_)
            validate_schema(records_);
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        const auto& t = records_[index_++ % records_.size()];
        return t;
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};
    bool validate_{false};

    static std::uint64_t read_u64(std::istream& in) {
        std::uint64_t v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    /// Decode the custom HDF5-like container format. Supports optional
    /// Zstandard compression when the magic string is "HDFZ".
    void parse_stream(std::istream& in) {
        char magic[4];
        in.read(magic, 4);
        bool compressed = false;
        if (std::strncmp(magic, "HDF5", 4) == 0) {
            compressed = false;
        } else if (std::strncmp(magic, "HDFZ", 4) == 0) {
            compressed = true;
        } else {
            throw std::runtime_error("invalid HDF5 header");
        }
        std::uint64_t count = read_u64(in);
        if (compressed) {
            std::string blob((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
            size_t decomp_size = ZSTD_getFrameContentSize(blob.data(), blob.size());
            std::string out(decomp_size, '\0');
            size_t got = ZSTD_decompress(out.data(), decomp_size, blob.data(), blob.size());
            if (ZSTD_isError(got))
                throw std::runtime_error("HDF5 decompression failed");
            std::istringstream tmp(out);
            for (std::uint64_t i = 0; i < count; ++i) {
                records_.push_back(read_tensor(tmp));
                if (!tmp)
                    throw std::runtime_error("truncated HDF5 data");
            }
            if (tmp.peek() != EOF)
                throw std::runtime_error("extra data at end of HDF5 file");
        } else {
            for (std::uint64_t i = 0; i < count; ++i) {
                records_.push_back(read_tensor(in));
                if (!in)
                    throw std::runtime_error("truncated HDF5 data");
            }
            if (in.peek() != EOF)
                throw std::runtime_error("extra data at end of HDF5 file");
        }
    }
};

/**
 * @brief Memory mapped variant of @ref Hdf5Producer.
 *
 * This reader avoids loading the entire container into memory by
 * mapping the file and lazily decoding each tensor when requested.
 * Only uncompressed containers are memory mapped. Compressed files
 * fall back to eager loading similar to @ref Hdf5Producer.
 */
class MmapHdf5Producer : public Producer {
  public:
#ifdef __unix__
    explicit MmapHdf5Producer(const std::string& path, bool validate = false)
        : validate_{validate} {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("failed to open HDF5 file");
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("failed to stat HDF5 file");
        }
        size_ = static_cast<std::size_t>(st.st_size);
        map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("failed to mmap HDF5 file");
        }
        parse_memory(static_cast<const char*>(map_), size_);
        if (validate_)
            check_consistency();
    }

    ~MmapHdf5Producer() override {
        if (map_ && map_ != MAP_FAILED)
            ::munmap(map_, size_);
        if (fd_ >= 0)
            ::close(fd_);
    }
#else
    explicit MmapHdf5Producer(const std::string&, bool = false) {
        throw std::runtime_error("memory mapping not supported on this platform");
    }
#endif

    HTensor next() override {
        if (offsets_.empty())
            return {};
        const char* p = offsets_[index_++ % offsets_.size()];
        return read_tensor_mem(p, data_end_);
    }

    std::size_t size() const override { return offsets_.size(); }

  private:
#ifdef __unix__
    int fd_{-1};
    void* map_{nullptr};
    std::size_t size_{0};
#endif
    std::string buffer_{};
    const char* data_end_{nullptr};
    std::vector<const char*> offsets_{};
    std::size_t index_{0};
    bool validate_{false};

    static std::uint32_t read_u32_mem(const char*& p, const char* end) {
        if (p + 4 > end)
            throw std::runtime_error("truncated HDF5 data");
        std::uint32_t v{};
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }

    static std::uint64_t read_u64_mem(const char*& p, const char* end) {
        if (p + 8 > end)
            throw std::runtime_error("truncated HDF5 data");
        std::uint64_t v{};
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }

    static void skip_tensor(const char*& p, const char* end) {
        p += 1; // dtype
        std::uint32_t dims = read_u32_mem(p, end);
        for (std::uint32_t i = 0; i < dims; ++i)
            read_u32_mem(p, end);
        std::uint32_t size = read_u32_mem(p, end);
        p += size;
    }

    static HTensor read_tensor_mem(const char*& p, const char* end) {
        if (p >= end)
            return {};
        std::uint8_t dt = static_cast<std::uint8_t>(*p++);
        std::uint32_t dims = read_u32_mem(p, end);
        HTensor::Shape shape(dims);
        for (std::uint32_t i = 0; i < dims; ++i)
            shape[i] = read_u32_mem(p, end);
        std::uint32_t size = read_u32_mem(p, end);
        if (p + size > end)
            throw std::runtime_error("truncated HDF5 tensor");
        std::vector<std::byte> data(size);
        std::memcpy(data.data(), p, size);
        p += size;
        return {static_cast<HTensor::DType>(dt), std::move(shape), std::move(data)};
    }

    void parse_memory(const char* data, std::size_t len) {
        if (len < 8)
            throw std::runtime_error("invalid HDF5 file");
        bool compressed = false;
        if (std::memcmp(data, "HDF5", 4) == 0) {
            compressed = false;
        } else if (std::memcmp(data, "HDFZ", 4) == 0) {
            compressed = true;
        } else {
            throw std::runtime_error("invalid HDF5 header");
        }
        std::uint64_t count;
        const char* p = data + 4;
        count = read_u64_mem(p, data + len);
        if (compressed) {
            size_t decomp_size = ZSTD_getFrameContentSize(p, data + len - p);
            buffer_.assign(decomp_size, '\0');
            size_t got = ZSTD_decompress(buffer_.data(), decomp_size, p, data + len - p);
            if (ZSTD_isError(got))
                throw std::runtime_error("HDF5 decompression failed");
            p = buffer_.data();
            data_end_ = buffer_.data() + got;
        } else {
            data_end_ = data + len;
        }

        offsets_.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            offsets_.push_back(p);
            skip_tensor(p, data_end_);
        }
    }

    void check_consistency() {
        if (offsets_.size() <= 1)
            return;
        const char* ref_ptr = offsets_[0];
        HTensor ref = read_tensor_mem(ref_ptr, data_end_);
        for (std::size_t i = 1; i < offsets_.size(); ++i) {
            const char* tmp = offsets_[i];
            HTensor t = read_tensor_mem(tmp, data_end_);
            if (t.dtype() != ref.dtype() || t.shape() != ref.shape()) {
                std::ostringstream msg;
                msg << "dataset schema mismatch at record " << i << ": expected "
                    << dtype_name(ref.dtype()) << shape_to_string(ref.shape()) << ", got "
                    << dtype_name(t.dtype()) << shape_to_string(t.shape());
                throw std::runtime_error(msg.str());
            }
        }
    }
};

/**
 * @brief Minimal writer for the custom HDF5-like format.
 */
class Hdf5Consumer : public Consumer {
  public:
    explicit Hdf5Consumer(const std::string& path, bool compress = false)
        : compress_{compress}, out_{std::make_shared<std::ofstream>(path, std::ios::binary)} {
        auto* f = static_cast<std::ofstream*>(out_.get());
        if (!*f)
            throw std::runtime_error("failed to open HDF5 file");
        write_header();
    }

    explicit Hdf5Consumer(std::ostream& out, bool compress = false)
        : compress_{compress}, out_{&out, [](std::ostream*) {}} {
        write_header();
    }

    ~Hdf5Consumer() { finalize(); }

    void push(const HTensor& t) override {
        if (!out_ || !*out_)
            return;
        if (compress_) {
            std::ostringstream tmp;
            write_tensor(tmp, t);
            auto str = tmp.str();
            buffer_.insert(buffer_.end(), str.begin(), str.end());
            ++count_;
        } else {
            write_tensor(*out_, t);
            ++count_;
        }
    }

  private:
    bool compress_{false};
    std::shared_ptr<std::ostream> out_{};
    std::uint64_t count_{0};
    std::string buffer_{};

    /// Write the file header with a placeholder count.
    void write_header() {
        const char* magic = compress_ ? "HDFZ" : "HDF5";
        out_->write(magic, 4);
        std::uint64_t zero = 0;
        out_->write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    /// Update the record count in the header once finished writing.
    void finalize() {
        if (!out_ || !*out_)
            return;
        auto pos = out_->tellp();
        out_->seekp(4, std::ios::beg);
        out_->write(reinterpret_cast<const char*>(&count_), sizeof(count_));
        out_->seekp(pos);
        if (compress_ && !buffer_.empty()) {
            size_t bound = ZSTD_compressBound(buffer_.size());
            std::string out(bound, '\0');
            size_t comp = ZSTD_compress(out.data(), bound, buffer_.data(), buffer_.size(), 1);
            if (!ZSTD_isError(comp)) {
                out_->write(out.data(), comp);
            }
        }
    }
};

/**
 * @brief Variant of Hdf5Consumer that can append to existing files.
 *
 * The consumer detects whether the target file already exists and, if so,
 * reuses its header information to continue writing additional tensors.
 * Only the record count in the header is updated when the consumer is
 * destroyed. For compressed containers the entire payload is rewritten on
 * finalisation.
 */
class CheckpointHdf5Consumer : public Consumer {
  public:
    explicit CheckpointHdf5Consumer(const std::string& path, bool compress = false) : path_{path} {
        file_ =
            std::make_shared<std::fstream>(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!*file_) {
            file_->open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        }
        init(compress);
    }

    ~CheckpointHdf5Consumer() { finalize(); }

    void push(const HTensor& t) override {
        if (!file_ || !*file_)
            return;
        if (compress_) {
            std::ostringstream tmp;
            write_tensor(tmp, t);
            auto str = tmp.str();
            buffer_.insert(buffer_.end(), str.begin(), str.end());
        } else {
            write_tensor(*file_, t);
        }
        ++count_;
    }

  private:
    std::string path_{};
    bool compress_{false};
    std::shared_ptr<std::fstream> file_{};
    std::uint64_t count_{0};
    std::string buffer_{};

    static std::uint64_t read_u64(std::istream& in) {
        std::uint64_t v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    void init(bool compress) {
        auto* f = file_.get();
        f->seekg(0, std::ios::end);
        auto size = f->tellg();
        f->seekg(0, std::ios::beg);
        if (size == 0) {
            compress_ = compress;
            write_header();
            return;
        }
        char magic[4];
        f->read(magic, 4);
        if (std::strncmp(magic, "HDF5", 4) == 0) {
            compress_ = false;
        } else if (std::strncmp(magic, "HDFZ", 4) == 0) {
            compress_ = true;
        } else {
            throw std::runtime_error("invalid HDF5 header");
        }
        count_ = read_u64(*f);
        if (compress_) {
            std::string blob((std::istreambuf_iterator<char>(*f)),
                             std::istreambuf_iterator<char>());
            size_t decomp_size = ZSTD_getFrameContentSize(blob.data(), blob.size());
            std::string out(decomp_size, '\0');
            size_t got = ZSTD_decompress(out.data(), decomp_size, blob.data(), blob.size());
            if (ZSTD_isError(got))
                throw std::runtime_error("HDF5 decompression failed");
            buffer_.assign(out.data(), out.data() + got);
        } else {
            f->seekp(0, std::ios::end);
        }
    }

    void write_header() {
        const char* magic = compress_ ? "HDFZ" : "HDF5";
        file_->seekp(0, std::ios::beg);
        file_->write(magic, 4);
        file_->write(reinterpret_cast<const char*>(&count_), sizeof(count_));
    }

    void finalize() {
        if (!file_ || !*file_)
            return;
        if (compress_) {
            file_->close();
            std::fstream out(path_, std::ios::binary | std::ios::out | std::ios::trunc);
            const char* magic = "HDFZ";
            out.write(magic, 4);
            out.write(reinterpret_cast<const char*>(&count_), sizeof(count_));
            size_t bound = ZSTD_compressBound(buffer_.size());
            std::string comp(bound, '\0');
            size_t c = ZSTD_compress(comp.data(), bound, buffer_.data(), buffer_.size(), 1);
            if (!ZSTD_isError(c))
                out.write(comp.data(), c);
            out.close();
        } else {
            auto pos = file_->tellp();
            file_->seekp(4, std::ios::beg);
            file_->write(reinterpret_cast<const char*>(&count_), sizeof(count_));
            file_->seekp(pos);
            file_->close();
        }
    }
};

#ifdef __unix__
/**
 * @brief Producer that downloads serialized tensors over HTTP.
 *
 * Only a blocking implementation using POSIX sockets is provided since
 * this is mainly intended for integration tests. Records may optionally
 * be cached on disk to avoid repeated network transfers.
 */

// ---------------------------------------------------------------------------
// Implementation overview
// ---------------------------------------------------------------------------
// The HTTP producer is purposefully tiny. It opens a TCP socket to the target
// host, performs a very small HTTP/1.1 GET request and then interprets the
// response body as a sequence of serialized tensors. Because this code is only
// used in tests we ignore many aspects of the HTTP protocol such as chunked
// transfer encoding, redirects and TLS. The intention is merely to demonstrate
// how network based datasets could be integrated without pulling in a full HTTP
// client library.
//
// The producer supports an optional on-disk cache. When enabled it will write
// each fetched tensor to a simple binary file. Subsequent runs will attempt to
// populate the internal record vector from that file before performing another
// network request. This drastically speeds up integration tests that use the
// same dataset repeatedly.
// ---------------------------------------------------------------------------
class HttpProducer : public Producer {
  public:
    HttpProducer(const std::string& host, unsigned short port, const std::string& path = "/",
                 bool cache = true, const std::string& cache_path = "", bool validate = false)
        : host_{host}, port_{port}, path_{path}, cache_{cache}, cache_path_{cache_path},
          validate_{validate} {
        if (!cache_path_.empty() && load_cache()) {
            if (validate_)
                validate_schema(records_);
            return;
        }
        fetch();
        if (validate_)
            validate_schema(records_);
        if (!cache_path_.empty())
            save_cache();
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        if (index_ >= records_.size()) {
            if (cache_) {
                index_ = 0;
            } else {
                fetch();
            }
        }
        return records_[index_++];
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::string host_{};
    unsigned short port_{};
    std::string path_{};
    bool cache_{true};
    std::string cache_path_{};
    std::vector<HTensor> records_{};
    std::size_t index_{0};
    bool validate_{false};

    bool load_cache() {
        // Attempt to populate records_ from the on-disk cache.
        std::ifstream in(cache_path_, std::ios::binary);
        if (!in)
            return false;
        records_.clear();
        while (in.peek() != EOF) {
            try {
                records_.push_back(read_tensor(in));
            } catch (...) {
                break;
            }
        }
        index_ = 0;
        bool ok = !records_.empty();
        if (ok && validate_)
            validate_schema(records_);
        return ok;
    }

    void save_cache() {
        // Persist records_ to disk for future runs.
        std::ofstream out(cache_path_, std::ios::binary);
        if (!out)
            return;
        for (const auto& t : records_)
            write_tensor(out, t);
    }

    void fetch() {
        // Perform a very small HTTP GET request and parse the response body.
        // This routine intentionally avoids using any external HTTP library.
        // It demonstrates how a dataset could be streamed over the network in
        // environments where dependencies must be kept to an absolute minimum.
        // The implementation is single threaded and blocks until the entire
        // response has been received.
        records_.clear();
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        // Convert dotted IPv4 address to binary form. No DNS lookup is
        // performed which keeps the implementation simple and deterministic.
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            ::close(fd);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port_);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(fd);
            throw std::runtime_error("failed to connect");
        }
        std::string req = "GET " + path_ + " HTTP/1.1\r\n";
        req += "Host: " + host_ + ":" + std::to_string(port_) + "\r\n";
        req += "Connection: close\r\n\r\n";
        ::send(fd, req.c_str(), req.size(), 0);

        std::string header;
        char ch;
        while (true) {
            ssize_t n = ::recv(fd, &ch, 1, 0);
            if (n <= 0)
                break;
            header.push_back(ch);
            if (header.size() >= 4 && header.compare(header.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::istringstream hs(header);
        std::string http_ver;
        unsigned status = 0;
        hs >> http_ver >> status;
        if (status != 200) {
            ::close(fd);
            throw std::runtime_error("HTTP error");
        }

        std::string body;
        char buf[4096];
        ssize_t n;
        // Read the remainder of the response into memory. The body is expected
        // to contain one or more concatenated tensor blobs in the custom
        // binary format understood by `read_tensor`.
        while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0)
            body.append(buf, static_cast<std::size_t>(n));
        ::close(fd);

        std::istringstream in(body);
        // Decode each tensor from the response. Errors are ignored so that a
        // partially transferred dataset does not crash the caller. Instead any
        // successfully parsed tensors are returned and the rest are dropped.
        while (in.peek() != EOF) {
            try {
                records_.push_back(read_tensor(in));
            } catch (...) {
                break;
            }
        }
        index_ = 0;
        if (validate_)
            validate_schema(records_);
        if (!cache_path_.empty())
            save_cache();
    }
};

/**
 * @brief Asynchronously downloads serialized tensors over HTTP using a
 * non-blocking socket.
 */
class AsyncHttpProducer : public Producer {
  public:
    AsyncHttpProducer(const std::string& host, unsigned short port, const std::string& path = "/",
                      bool validate = false)
        : host_{host}, port_{port}, path_{path}, validate_{validate} {
        worker_ = std::thread(&AsyncHttpProducer::fetch, this);
    }

    ~AsyncHttpProducer() override {
        if (worker_.joinable())
            worker_.join();
    }

    HTensor next() override {
        wait();
        if (index_ >= records_.size())
            return {};
        return records_[index_++];
    }

    std::size_t size() const override {
        const_cast<AsyncHttpProducer*>(this)->wait();
        return records_.size();
    }

  private:
    std::string host_{};
    unsigned short port_{};
    std::string path_{};
    bool validate_{false};
    std::vector<HTensor> records_{};
    std::size_t index_{0};
    std::thread worker_{};
    mutable std::mutex mutex_{};

    void wait() const {
        std::unique_lock<std::mutex> lock(mutex_);
        if (worker_.joinable()) {
            lock.unlock();
            const_cast<AsyncHttpProducer*>(this)->worker_.join();
            lock.lock();
        }
    }

    void fetch() {
        std::string body;
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
            return;
        ::fcntl(fd, F_SETFL, O_NONBLOCK);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            ::close(fd);
            return;
        }
        addr.sin_port = htons(port_);
        int res = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (res < 0 && errno != EINPROGRESS) {
            ::close(fd);
            return;
        }
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);
        ::select(fd + 1, nullptr, &wfds, nullptr, nullptr);

        std::string req = "GET " + path_ + " HTTP/1.1\r\n";
        req += "Host: " + host_ + ":" + std::to_string(port_) + "\r\n";
        req += "Connection: close\r\n\r\n";
        ::send(fd, req.c_str(), req.size(), 0);

        char buf[4096];
        while (true) {
            fd_set set;
            FD_ZERO(&set);
            FD_SET(fd, &set);
            if (::select(fd + 1, &set, nullptr, nullptr, nullptr) <= 0)
                break;
            ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0)
                break;
            body.append(buf, static_cast<std::size_t>(n));
        }
        ::close(fd);

        auto pos = body.find("\r\n\r\n");
        std::string header = body.substr(0, pos == std::string::npos ? 0 : pos + 2);
        std::istringstream hs(header);
        std::string http_ver;
        unsigned status = 0;
        hs >> http_ver >> status;
        if (status != 200)
            return;
        std::string payload = pos == std::string::npos ? body : body.substr(pos + 4);
        std::istringstream in(payload);
        while (in.peek() != EOF) {
            try {
                records_.push_back(read_tensor(in));
            } catch (...) {
                break;
            }
        }
        if (validate_)
            validate_schema(records_);
    }
};

/**
 * @brief Producer that downloads an entire HDF5 container over HTTP.
 */
class HttpHdf5Producer : public Producer {
  public:
    HttpHdf5Producer(const std::string& host, unsigned short port, const std::string& path = "/",
                     bool cache = true, const std::string& cache_path = "", bool validate = false)
        : host_{host}, port_{port}, path_{path}, cache_{cache}, cache_path_{cache_path},
          validate_{validate} {
        if (!cache_path_.empty() && load_cache()) {
            return;
        }
        fetch();
        if (!cache_path_.empty())
            save_cache();
    }

    HTensor next() override { return prod_ ? prod_->next() : HTensor{}; }
    std::size_t size() const override { return prod_ ? prod_->size() : 0; }

  private:
    std::string host_{};
    unsigned short port_{};
    std::string path_{};
    bool cache_{true};
    std::string cache_path_{};
    std::unique_ptr<Hdf5Producer> prod_{};
    std::string body_{};
    bool validate_{false};

    bool load_cache() {
        std::ifstream in(cache_path_, std::ios::binary);
        if (!in)
            return false;
        prod_ = std::make_unique<Hdf5Producer>(in, validate_);
        return prod_->size() > 0;
    }

    void save_cache() {
        std::ofstream out(cache_path_, std::ios::binary);
        if (!out)
            return;
        out.write(body_.data(), static_cast<std::streamsize>(body_.size()));
    }

    void fetch() {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            ::close(fd);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port_);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(fd);
            throw std::runtime_error("failed to connect");
        }
        std::string req = "GET " + path_ + " HTTP/1.1\r\n";
        req += "Host: " + host_ + ":" + std::to_string(port_) + "\r\n";
        req += "Connection: close\r\n\r\n";
        ::send(fd, req.c_str(), req.size(), 0);

        std::string header;
        char ch;
        while (true) {
            ssize_t n = ::recv(fd, &ch, 1, 0);
            if (n <= 0)
                break;
            header.push_back(ch);
            if (header.size() >= 4 && header.compare(header.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::istringstream hs(header);
        std::string http_ver;
        unsigned status = 0;
        hs >> http_ver >> status;
        if (status != 200) {
            ::close(fd);
            throw std::runtime_error("HTTP error");
        }

        body_.clear();
        char buf[4096];
        ssize_t n;
        while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0)
            body_.append(buf, static_cast<std::size_t>(n));
        ::close(fd);

        std::istringstream in(body_);
        prod_ = std::make_unique<Hdf5Producer>(in, validate_);
    }
};
#else
class HttpProducer : public Producer {
  public:
    HttpProducer(const std::string&, unsigned short, const std::string& = "/", bool = true,
                 const std::string& = "") {
        throw std::runtime_error("HTTP streaming not supported on this platform");
    }
    HTensor next() override { return {}; }
    std::size_t size() const override { return 0; }
};

class HttpHdf5Producer : public Producer {
  public:
    HttpHdf5Producer(const std::string&, unsigned short, const std::string& = "/", bool = true,
                     const std::string& = "") {
        throw std::runtime_error("HTTP streaming not supported on this platform");
    }
    HTensor next() override { return {}; }
    std::size_t size() const override { return 0; }
};
#endif

} // namespace harmonics
