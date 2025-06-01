#include <cmath>
#include <fstream>
#include <gpu/GPUFunction.hpp>
#include <gpu/GlobalFunctionRegistry.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace harmonics;

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("failed to open input file: " + path);
    std::vector<float> vals;
    float v;
    while (in >> v)
        vals.push_back(v);
    return vals;
}

struct SimpleWrapper {
    explicit SimpleWrapper(std::string name) : shader(std::move(name)) {}
    GPUDataVariant run(const std::vector<GPUDataVariant>& inputs) const {
        const auto* fn = GPUFunctionRegistry::getInstance().get(shader);
        if (!fn)
            throw std::runtime_error("shader not found");
        return fn->cpuFallback(inputs);
    }
    std::string shader;
};

int main(int argc, char** argv) {
    std::string shader;
    std::vector<std::string> inputs;
    std::string out_path = "shader_output.txt";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shader" && i + 1 < argc) {
            shader = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            inputs.emplace_back(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            out_path = argv[++i];
        }
    }
    if (shader.empty() || inputs.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --shader NAME --input file [--input file ...] [-o output]" << std::endl;
        return 1;
    }

    registerAllShaders();

    std::vector<std::vector<float>> data;
    for (const auto& file : inputs)
        data.push_back(read_floats(file));

    std::vector<GPUDataVariant> params;
    for (const auto& v : data)
        params.emplace_back(v);

    SimpleWrapper wrapper(shader);
    GPUDataVariant result = wrapper.run(params);

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "failed to open output file" << std::endl;
        return 1;
    }

    out << "Shader: " << shader << '\n';
    for (size_t i = 0; i < data.size(); ++i) {
        out << "Input" << i << ':';
        for (float v : data[i])
            out << ' ' << v;
        out << '\n';
    }
    out << "Output:";
    if (auto p = std::get_if<std::vector<float>>(&result)) {
        for (float v : *p)
            out << ' ' << v;
    } else if (auto p2 = std::get_if<float>(&result)) {
        out << ' ' << *p2;
    }
    out << '\n';
    return 0;
}
