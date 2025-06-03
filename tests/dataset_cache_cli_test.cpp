#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#ifdef __unix__
#include <sys/wait.h>
#endif

#include <harmonics/dataset.hpp>
#include <harmonics/dataset_cache.hpp>
#include <harmonics/distributed_io.hpp>

using namespace harmonics;

namespace {

struct CountingProducer : Producer {
    explicit CountingProducer(int n) : limit(n) {}
    HTensor next() override {
        if (index >= limit)
            return {};
        float v = static_cast<float>(index++);
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return HTensor{HTensor::DType::Float32, {1}, std::move(data)};
    }
    std::size_t size() const override { return limit; }
    int limit;
    int index{0};
};

static HTensor make_tensor(float v) {
    HTensor t{HTensor::DType::Float32, {1}};
    t.data().resize(sizeof(float));
    std::memcpy(t.data().data(), &v, sizeof(float));
    return t;
}

void create_hdf5(const char* path) {
    Hdf5Consumer cons(path);
    cons.push(make_tensor(1.f));
    cons.push(make_tensor(2.f));
}

} // namespace

#ifdef __unix__
TEST(DatasetCacheCli, Download) {
    const char* src = "download_src.h5";
    const char* dst = "download_dst.h5";
    create_hdf5(src);

    SocketServer server;
    std::thread srv([&]() {
        auto c = server.accept_consumer();
        Hdf5Producer prod(src);
        for (std::size_t i = 0; i < prod.size(); ++i)
            c->push(prod.next());
        c->push({});
    });

    std::string cmd = std::string("./dataset_cache_cli download ") + dst + " 127.0.0.1 " +
                      std::to_string(server.port()) + " > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    srv.join();

    Hdf5Producer check(dst);
    EXPECT_EQ(check.size(), 2u);
    auto a = check.next();
    auto b = check.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 1.f);
    EXPECT_EQ(db[0], 2.f);
    std::remove(src);
    std::remove(dst);
}

TEST(DatasetCacheCli, Upload) {
    const char* src = "upload_src.h5";
    create_hdf5(src);

    SocketServer server;
    std::vector<HTensor> received;
    std::thread srv([&]() {
        auto p = server.accept_producer();
        received.push_back(p->next());
        received.push_back(p->next());
        received.push_back(p->next());
    });

    std::string cmd = std::string("./dataset_cache_cli upload ") + src + " 127.0.0.1 " +
                      std::to_string(server.port()) + " > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    srv.join();

    ASSERT_EQ(received.size(), 3u);
    const float* d1 = reinterpret_cast<const float*>(received[0].data().data());
    const float* d2 = reinterpret_cast<const float*>(received[1].data().data());
    EXPECT_EQ(d1[0], 1.f);
    EXPECT_EQ(d2[0], 2.f);
    EXPECT_EQ(received[2].data().size(), 0u);
    std::remove(src);
}

TEST(DatasetCacheCli, ServeDownloadAndClient) {
    const char* src = "serve_dl_src.h5";
    const char* dst = "serve_dl_dst.h5";
    create_hdf5(src);

    FILE* pipe = popen("./dataset_cache_cli serve-download serve_dl_src.h5", "r");
    ASSERT_EQ(pipe != nullptr, true);
    unsigned short port = 0;
    ASSERT_EQ(fscanf(pipe, "%hu", &port), 1);

    std::string cmd = std::string("./dataset_cache_cli download ") + dst + " 127.0.0.1 " +
                      std::to_string(port) + " > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);

    int status = pclose(pipe);
    ASSERT_EQ(WIFEXITED(status), true);
    ASSERT_EQ(WEXITSTATUS(status), 0);

    Hdf5Producer check(dst);
    EXPECT_EQ(check.size(), 2u);
    std::remove(src);
    std::remove(dst);
}

TEST(DatasetCacheCli, ServeUploadAndClient) {
    const char* src = "serve_ul_src.h5";
    const char* dst = "serve_ul_dst.h5";
    create_hdf5(src);

    FILE* pipe = popen("./dataset_cache_cli serve-upload serve_ul_dst.h5", "r");
    ASSERT_EQ(pipe != nullptr, true);
    unsigned short port = 0;
    ASSERT_EQ(fscanf(pipe, "%hu", &port), 1);

    std::string cmd = std::string("./dataset_cache_cli upload ") + src + " 127.0.0.1 " +
                      std::to_string(port) + " > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);

    int status = pclose(pipe);
    ASSERT_EQ(WIFEXITED(status), true);
    ASSERT_EQ(WEXITSTATUS(status), 0);

    Hdf5Producer check(dst);
    EXPECT_EQ(check.size(), 2u);
    std::remove(src);
    std::remove(dst);
}

TEST(DatasetCacheCli, ResumeDownload) {
    const char* src = "resume_src.h5";
    const char* dst = "resume_dst.h5";
    create_hdf5(src);

    // Simulate an interrupted transfer by creating a partial cache with a
    // single record already present locally.
    {
        Hdf5Consumer partial_cons(dst);
        partial_cons.push(make_tensor(1.f));
    }

    // Resume the download to fetch the remaining records.
    SocketServer full;
    std::thread srv2([&]() {
        auto c = full.accept_consumer();
        Hdf5Producer prod(src);
        for (std::size_t i = 0; i < prod.size(); ++i)
            c->push(prod.next());
        c->push({});
    });

    std::string cmd = std::string("./dataset_cache_cli download ") + dst + " 127.0.0.1 " +
                      std::to_string(full.port()) + " > /dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    srv2.join();

    Hdf5Producer check(dst);
    EXPECT_EQ(check.size(), 2u);

    std::remove(src);
    std::remove(dst);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
