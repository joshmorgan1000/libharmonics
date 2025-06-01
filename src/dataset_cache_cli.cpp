#include <blake3.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <harmonics/dataset_cache.hpp>
#include <harmonics/distributed_io.hpp>
#include <harmonics/tcp_io.hpp>

// ---------------------------------------------------------------------------
// Dataset cache CLI overview
// ---------------------------------------------------------------------------
// This small program exposes a handful of commands for synchronising cached
// datasets between machines. It can act as a client or a tiny server using
// the TCP based producer/consumer helpers found in the library. The intent is
// to make sharing preprocessed datasets as lightweight as possible during
// development. Error handling is minimal so unexpected network failures will
// simply surface as exceptions.
// ---------------------------------------------------------------------------

using namespace harmonics;

namespace {

void usage() {
    std::cerr << "Usage:\n"
              << "  dataset_cache_cli download <path> <host> <port>\n"
              << "  dataset_cache_cli upload <path> <host> <port>\n"
              << "  dataset_cache_cli serve-download <path> [port]\n"
              << "  dataset_cache_cli serve-upload <path> [port]\n"
              << "  dataset_cache_cli hash <path>\n";
}

std::string hash_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open file");
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    char buf[4096];
    // Stream the file in small chunks so very large caches do not
    // consume excessive memory while computing the hash.
    while (in.good()) {
        in.read(buf, sizeof(buf));
        std::streamsize got = in.gcount();
        if (got > 0)
            blake3_hasher_update(&hasher, reinterpret_cast<const uint8_t*>(buf),
                                 static_cast<size_t>(got));
    }
    uint8_t out[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    return harmonics::to_hex(out, BLAKE3_OUT_LEN);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];

    try {
        if (cmd == "download" && argc >= 5) {
            // Retrieve a cache from a remote host over TCP. If the file already
            // exists the transfer resumes by skipping any tensors that have been
            // written previously.
            std::string path = argv[2];
            std::string host = argv[3];
            unsigned short port = static_cast<unsigned short>(std::stoi(argv[4]));

            std::size_t existing = 0;
            {
                std::ifstream chk(path, std::ios::binary);
                if (chk) {
                    Hdf5Producer tmp(path);
                    existing = tmp.size();
                }
            }

            auto remote = std::make_shared<TcpProducer>(host, port);
            for (std::size_t i = 0; i < existing; ++i) {
                HTensor skip = remote->next();
                if (skip.data().empty())
                    return 0;
            }

            if (existing == 0) {
                CachedProducer cache(nullptr, path);
                cache.download(remote);
            } else {
                CheckpointHdf5Consumer writer(path);
                while (true) {
                    HTensor t = remote->next();
                    if (t.data().empty())
                        break;
                    writer.push(t);
                }
            }

            return 0;
        }
        if (cmd == "upload" && argc >= 5) {
            // Send the local cache to a remote host.
            std::string path = argv[2];
            std::string host = argv[3];
            unsigned short port = static_cast<unsigned short>(std::stoi(argv[4]));
            CachedProducer cache(nullptr, path);
            auto remote = std::make_shared<TcpConsumer>(host, port);
            cache.upload(remote);
            return 0;
        }
        if (cmd == "serve-download" && argc >= 3) {
            // Act as a temporary server so other machines can fetch our cache.
            std::string path = argv[2];
            unsigned short port = argc >= 4 ? static_cast<unsigned short>(std::stoi(argv[3])) : 0;
            SocketServer server(port);
            std::cout << server.port() << std::endl;
            auto cons = server.accept_consumer();
            CachedProducer cache(nullptr, path);
            cache.upload(std::shared_ptr<Consumer>(std::move(cons)));
            return 0;
        }
        if (cmd == "serve-upload" && argc >= 3) {
            // Host a producer so remote machines can push their cache to us.
            std::string path = argv[2];
            unsigned short port = argc >= 4 ? static_cast<unsigned short>(std::stoi(argv[3])) : 0;
            SocketServer server(port);
            std::cout << server.port() << std::endl;
            auto prod = server.accept_producer();
            CachedProducer cache(nullptr, path);
            cache.download(std::shared_ptr<Producer>(std::move(prod)));
            return 0;
        }
        if (cmd == "hash" && argc >= 3) {
            // Calculate and print a BLAKE3 digest of the cache file.
            std::string path = argv[2];
            std::string digest = hash_file(path);
            std::cout << digest << std::endl;
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    // If no command matched print the usage information so the
    // caller knows how to invoke the tool correctly.
    usage();
    return 1;
}
