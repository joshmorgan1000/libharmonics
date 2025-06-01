#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <harmonics/dataset.hpp>
#include <harmonics/schema_validation.hpp>

// ---------------------------------------------------------------------------
// Dataset schema validator
// ---------------------------------------------------------------------------
// Simple command line helper that checks whether a dataset matches the
// specified shape and element type. It does this by wrapping an existing
// producer in a SchemaValidatingProducer and reporting how many samples could
// be read without violating the constraints.
// ---------------------------------------------------------------------------

using namespace harmonics;

namespace {

void usage() {
    std::cerr << "Usage:\n"
              << "  dataset_schema_cli <path> <dtype> <dim1> [dim2 ...]\n";
}

HTensor::DType parse_dtype(const std::string& s) {
    // Map the short string tokens used on the command line to the
    // internal HTensor::DType enumeration. Only the handful of types
    // required by the unit tests are supported.
    if (s == "f32")
        return HTensor::DType::Float32;
    if (s == "f64")
        return HTensor::DType::Float64;
    if (s == "i32")
        return HTensor::DType::Int32;
    if (s == "i64")
        return HTensor::DType::Int64;
    if (s == "u8")
        return HTensor::DType::UInt8;
    throw std::runtime_error("unknown dtype");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }
    std::string path = argv[1];
    std::string dtype_str = argv[2];
    HTensor::Shape shape;
    // Remaining arguments describe the expected tensor dimensions. Each
    // value is converted to a size_t and appended to the shape vector.
    for (int i = 3; i < argc; ++i)
        shape.push_back(static_cast<std::size_t>(std::stoul(argv[i])));

    try {
        // Wrap the dataset with a schema validating producer so that
        // each tensor is checked against the requested type and shape.
        auto dtype = parse_dtype(dtype_str);
        auto base = std::make_shared<Hdf5Producer>(path);
        SchemaValidatingProducer validator{base, shape, dtype, true};
        std::cout << "Validated " << validator.size() << " samples" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
