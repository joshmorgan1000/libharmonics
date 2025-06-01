#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <harmonics/dataset.hpp>

// ---------------------------------------------------------------------------
// Dataset conversion utility
// ---------------------------------------------------------------------------
// Converts trivial CSV or IDX formatted data into the lightweight HDF5 based
// container used throughout the unit tests. The implementation focuses on
// clarity over performance and therefore reads the entire dataset into memory
// before writing the output file.
// ---------------------------------------------------------------------------

using namespace harmonics;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: dataset_convert (--csv|--idx) <in> [-o out.hdf5]\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string path = argv[2];
    std::string out = "out.hdf5";
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        // Allow the output path to be customised via -o/--out.  Any
        // additional arguments are simply ignored to keep the tool
        // minimal and focused on conversion.
        if ((arg == "-o" || arg == "--out") && i + 1 < argc)
            out = argv[++i];
    }

    try {
        // Select the appropriate parser based on the command line flag.
        std::shared_ptr<Producer> prod;
        if (mode == "--csv") {
            prod = std::make_shared<CsvProducer>(path);
        } else if (mode == "--idx") {
            prod = std::make_shared<IdxProducer>(path);
        } else {
            std::cerr << "Unknown mode\n";
            return 1;
        }

        // Read all samples from the chosen producer and write them to
        // an HDF5 container. The consumer handles file creation and
        // compression transparently so the loop here is trivial.
        Hdf5Consumer writer(out);
        for (std::size_t i = 0; i < prod->size(); ++i)
            writer.push(prod->next());

        // HDF5Consumer signals completion when destructed so there is
        // no explicit close step required here. Errors will propagate
        // via exceptions making the tool fail fast on IO issues.
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
