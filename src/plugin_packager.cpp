#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {
void usage() {
    std::cerr << "Usage:\n";
    std::cerr << "  plugin_packager package <directory> <archive>\n";
    std::cerr << "  plugin_packager install <archive> <directory>\n";
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "package" && argc >= 4) {
            fs::path dir = argv[2];
            std::string archive = argv[3];
            if (!fs::exists(dir)) {
                std::cerr << "Directory not found\n";
                return 1;
            }
            std::string command = "tar --zstd -cf " + archive + " -C " + dir.string() + " .";
            int res = std::system(command.c_str());
            return res == 0 ? 0 : 1;
        } else if (cmd == "install" && argc >= 4) {
            std::string archive = argv[2];
            fs::path dir = argv[3];
            fs::create_directories(dir);
            std::string command = "tar --zstd -xf " + archive + " -C " + dir.string();
            int res = std::system(command.c_str());
            return res == 0 ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    usage();
    return 1;
}
