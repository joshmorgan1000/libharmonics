#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <harmonics/model_import.hpp>
#include <harmonics/serialization.hpp>

using namespace harmonics;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: model_convert (--onnx|--tensorflow|--pytorch) <in> [-o out]\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string path = argv[2];
    std::string out = "weights.hnwt";
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--out") && i + 1 < argc)
            out = argv[++i];
    }

    try {
        std::vector<NamedTensor> weights;
        if (mode == "--onnx") {
            weights = import_onnx_weights(path);
        } else if (mode == "--tensorflow") {
            weights = import_tensorflow_weights(path);
        } else if (mode == "--pytorch") {
            weights = import_pytorch_weights(path);
        } else {
            std::cerr << "Unknown mode\n";
            return 1;
        }
        std::ofstream ofs(out, std::ios::binary);
        if (!ofs) {
            std::cerr << "failed to open output file\n";
            return 1;
        }
        save_named_weights(weights, ofs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
