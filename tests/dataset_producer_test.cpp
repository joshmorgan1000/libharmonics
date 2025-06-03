#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <harmonics/dataset.hpp>
#include <harmonics/dataset_cache.hpp>
#include <harmonics/schema_validation.hpp>
#include <harmonics/serialization.hpp>
#include <harmonics/stream_io.hpp>
#ifdef __unix__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

struct CountingProducer : harmonics::Producer {
    explicit CountingProducer(int n) : limit{n} {}
    harmonics::HTensor next() override {
        if (index >= limit)
            return {};
        float v = static_cast<float>(index++);
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}, std::move(data)};
    }
    std::size_t size() const override { return limit; }
    int limit;
    int index{0};
};

TEST(DatasetProducerTest, CsvProducerParsesRows) {
    std::string csv = "1,2,3\n4,5,6\n";
    std::istringstream in(csv);
    harmonics::CsvProducer prod(in);

    EXPECT_EQ(prod.size(), 2u);

    auto t1 = prod.next();
    EXPECT_EQ(t1.shape().size(), 1u);
    EXPECT_EQ(t1.shape()[0], 3u);
    const float* d1 = reinterpret_cast<const float*>(t1.data().data());
    EXPECT_EQ(d1[0], 1.0f);
    EXPECT_EQ(d1[1], 2.0f);
    EXPECT_EQ(d1[2], 3.0f);

    auto t2 = prod.next();
    const float* d2 = reinterpret_cast<const float*>(t2.data().data());
    EXPECT_EQ(d2[0], 4.0f);
    EXPECT_EQ(d2[1], 5.0f);
    EXPECT_EQ(d2[2], 6.0f);
}

TEST(DatasetProducerTest, CsvProducerHandlesQuotes) {
    std::string csv = "\"1\", 2 ,\"3\"\n";
    std::istringstream in(csv);
    harmonics::CsvProducer prod(in);
    ASSERT_EQ(prod.size(), 1u);
    auto t = prod.next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 1.0f);
    EXPECT_EQ(d[1], 2.0f);
    EXPECT_EQ(d[2], 3.0f);
}

TEST(DatasetProducerTest, StreamingCsvProducerStreamsRows) {
    std::string csv = "10,20\n30,40\n";
    std::istringstream in(csv);
    harmonics::StreamingCsvProducer prod(in);

    auto a = prod.next();
    auto b = prod.next();
    auto c = prod.next();

    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());

    EXPECT_EQ(a.shape()[0], 2u);
    EXPECT_EQ(da[0], 10.f);
    EXPECT_EQ(db[1], 40.f);
    EXPECT_EQ(c.data().size(), 0u);
}

TEST(DatasetProducerTest, IdxProducerParsesData) {
    std::string bytes;
    auto append_be32 = [&](uint32_t v) {
        for (int i = 3; i >= 0; --i)
            bytes.push_back(static_cast<char>((v >> (i * 8)) & 0xFF));
    };
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(static_cast<char>(0x08));
    bytes.push_back(2); // dims
    append_be32(2);     // records
    append_be32(2);     // elements
    bytes.push_back(1);
    bytes.push_back(2);
    bytes.push_back(3);
    bytes.push_back(4);

    std::istringstream in(bytes);
    harmonics::IdxProducer prod(in);

    EXPECT_EQ(prod.size(), 2u);
    auto t1 = prod.next();
    EXPECT_EQ(t1.dtype(), harmonics::HTensor::DType::UInt8);
    EXPECT_EQ(t1.shape().size(), 1u);
    EXPECT_EQ(t1.shape()[0], 2u);
    const std::byte* d1 = t1.data().data();
    EXPECT_EQ(static_cast<unsigned>(d1[0]), 1u);
    EXPECT_EQ(static_cast<unsigned>(d1[1]), 2u);

    auto t2 = prod.next();
    const std::byte* d2 = t2.data().data();
    EXPECT_EQ(static_cast<unsigned>(d2[0]), 3u);
    EXPECT_EQ(static_cast<unsigned>(d2[1]), 4u);
}

TEST(DatasetProducerTest, IdxProducerParsesInt32) {
    std::string bytes;
    auto append_be32 = [&](uint32_t v) {
        for (int i = 3; i >= 0; --i)
            bytes.push_back(static_cast<char>((v >> (i * 8)) & 0xFF));
    };
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(static_cast<char>(0x0C));
    bytes.push_back(2); // dims
    append_be32(2);     // records
    append_be32(2);     // elements
    append_be32(1000);
    append_be32(2000);
    append_be32(3000);
    append_be32(4000);

    std::istringstream in(bytes);
    harmonics::IdxProducer prod(in);

    EXPECT_EQ(prod.size(), 2u);
    auto t1 = prod.next();
    EXPECT_EQ(t1.dtype(), harmonics::HTensor::DType::Int32);
    const int32_t* d1 = reinterpret_cast<const int32_t*>(t1.data().data());
    EXPECT_EQ(d1[0], 1000);
    EXPECT_EQ(d1[1], 2000);
    auto t2 = prod.next();
    const int32_t* d2 = reinterpret_cast<const int32_t*>(t2.data().data());
    EXPECT_EQ(d2[0], 3000);
    EXPECT_EQ(d2[1], 4000);
}

TEST(DatasetProducerTest, ShuffleProducerRandomises) {
    auto base = std::make_shared<CountingProducer>(4);
    harmonics::ShuffleProducer shuf{base};
    std::vector<int> values;
    for (int i = 0; i < 4; ++i) {
        auto t = shuf.next();
        auto d = reinterpret_cast<const float*>(t.data().data());
        values.push_back(static_cast<int>(d[0]));
    }
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[3], 3);
}

TEST(DatasetProducerTest, BatchProducerCombinesSamples) {
    auto base = std::make_shared<CountingProducer>(4);
    harmonics::BatchProducer batch{base, 2};
    auto t = batch.next();
    EXPECT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 1u);
    auto d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 0.0f);
    EXPECT_EQ(d[1], 1.0f);
}

TEST(DatasetProducerTest, AugmentProducerTransforms) {
    auto base = std::make_shared<CountingProducer>(1);
    harmonics::AugmentProducer aug{
        base, [](const harmonics::HTensor& t) {
            auto d = reinterpret_cast<const float*>(t.data().data());
            float v = d[0] * 2.0f;
            std::vector<std::byte> data(sizeof(float));
            std::memcpy(data.data(), &v, sizeof(float));
            return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}, std::move(data)};
        }};
    auto t = aug.next();
    auto d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 0.0f * 2.0f);
}

TEST(DatasetProducerTest, ShardProducerDividesDataset) {
    auto base = std::make_shared<CountingProducer>(5);
    harmonics::ShardProducer shard{base, 1, 2};
    EXPECT_EQ(shard.size(), 2u);
    auto a = shard.next();
    auto b = shard.next();
    auto d1 = reinterpret_cast<const float*>(a.data().data());
    auto d2 = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(d1[0], 3.f);
    EXPECT_EQ(d2[0], 4.f);
}

TEST(DatasetProducerTest, Hdf5ProducerRoundtrip) {
    const char* path = "hdf5_test.bin";
    {
        harmonics::Hdf5Consumer cons(path);
        CountingProducer base(1);
        cons.push(base.next());
        cons.push(base.next());
    }
    harmonics::Hdf5Producer prod(path);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 0.f);
    EXPECT_EQ(db[0], 1.f);
    std::remove(path);
}

TEST(DatasetProducerTest, Hdf5CompressedRoundtrip) {
    const char* path = "hdf5_comp.bin";
    {
        harmonics::Hdf5Consumer cons(path, true);
        CountingProducer base(2);
        cons.push(base.next());
        cons.push(base.next());
    }
    harmonics::Hdf5Producer prod(path);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 0.f);
    EXPECT_EQ(db[0], 1.f);
    std::remove(path);
}

TEST(DatasetProducerTest, CheckpointHdf5ConsumerAppends) {
    const char* path = "hdf5_checkpoint.bin";
    {
        harmonics::Hdf5Consumer cons(path);
        CountingProducer base(1);
        cons.push(base.next());
    }
    {
        harmonics::CheckpointHdf5Consumer cons(path);
        CountingProducer base(2);
        base.next();
        cons.push(base.next());
    }
    harmonics::Hdf5Producer prod(path);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 0.f);
    EXPECT_EQ(db[0], 1.f);
    std::remove(path);
}

TEST(DatasetProducerTest, TFRecordProducerParsesRecords) {
    std::string bytes;
    auto add_rec = [&](const std::string& s) {
        std::uint64_t len = s.size();
        bytes.append(reinterpret_cast<const char*>(&len), sizeof(len));
        std::uint32_t crc = 0;
        bytes.append(reinterpret_cast<const char*>(&crc), sizeof(crc));
        bytes.append(s);
        bytes.append(reinterpret_cast<const char*>(&crc), sizeof(crc));
    };
    add_rec("abc");
    add_rec("de");
    std::istringstream in(bytes);
    harmonics::TFRecordProducer prod(in);
    EXPECT_EQ(prod.size(), 2u);
    auto t = prod.next();
    EXPECT_EQ(t.shape()[0], 3u);
}

TEST(DatasetProducerTest, CocoJsonProducerParsesBBoxes) {
    std::string json = "{\"annotations\":[{\"bbox\":[1,2,3,4]},{\"bbox\":[5,6,7,8]}]}";
    std::istringstream in(json);
    harmonics::CocoJsonProducer prod(in);
    EXPECT_EQ(prod.size(), 2u);
    auto t = prod.next();
    EXPECT_EQ(t.shape()[0], 4u);
}

TEST(DatasetProducerTest, SchemaValidatingProducerPasses) {
    std::string csv = "1,2\n3,4\n";
    std::istringstream in(csv);
    auto base = std::make_shared<harmonics::CsvProducer>(in);
    harmonics::SchemaValidatingProducer val{base, {2}, harmonics::HTensor::DType::Float32};
    auto t = val.next();
    EXPECT_EQ(t.shape()[0], 2u);
}

TEST(DatasetProducerTest, SchemaValidatingProducerThrows) {
    std::string csv = "5,6\n";
    std::istringstream in(csv);
    auto base = std::make_shared<harmonics::CsvProducer>(in);
    bool threw = false;
    try {
        harmonics::SchemaValidatingProducer tmp{base, {3}, harmonics::HTensor::DType::Float32};
        (void)tmp;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

TEST(DatasetProducerTest, CachedProducerStoresAndReloads) {
    const char* path = "cached_dataset.bin";
    {
        auto base = std::make_shared<CountingProducer>(2);
        harmonics::CachedProducer cached(base, path);
        auto a = cached.next();
        auto b = cached.next();
        const float* da = reinterpret_cast<const float*>(a.data().data());
        const float* db = reinterpret_cast<const float*>(b.data().data());
        EXPECT_EQ(da[0], 0.f);
        EXPECT_EQ(db[0], 1.f);
    }

    harmonics::CachedProducer from_cache(nullptr, path);
    EXPECT_EQ(from_cache.size(), 2u);
    auto r1 = from_cache.next();
    auto r2 = from_cache.next();
    const float* d1 = reinterpret_cast<const float*>(r1.data().data());
    const float* d2 = reinterpret_cast<const float*>(r2.data().data());
    EXPECT_EQ(d1[0], 0.f);
    EXPECT_EQ(d2[0], 1.f);
    std::remove(path);
}

TEST(DatasetProducerTest, CachedProducerDownloadsFromRemote) {
    const char* path = "cache_download.bin";
    auto remote = std::make_shared<CountingProducer>(2);
    harmonics::CachedProducer cached(nullptr, path);
    cached.download(remote);

    EXPECT_EQ(cached.size(), 2u);
    auto a = cached.next();
    auto b = cached.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 0.f);
    EXPECT_EQ(db[0], 1.f);
    std::remove(path);
}

TEST(DatasetProducerTest, CachedProducerUploadsToRemote) {
    const char* path = "cache_upload.bin";
    {
        auto base = std::make_shared<CountingProducer>(2);
        harmonics::CachedProducer tmp(base, path);
        tmp.next();
        tmp.next();
    }

    auto bus = std::make_shared<harmonics::MessageBus>();
    auto cons = std::make_shared<harmonics::BusConsumer>(bus);
    harmonics::CachedProducer cached(nullptr, path);
    cached.upload(cons);

    harmonics::BusProducer prod(bus);
    auto r1 = prod.next();
    auto r2 = prod.next();
    auto done = prod.next();
    const float* d1 = reinterpret_cast<const float*>(r1.data().data());
    const float* d2 = reinterpret_cast<const float*>(r2.data().data());
    EXPECT_EQ(d1[0], 0.f);
    EXPECT_EQ(d2[0], 1.f);
    EXPECT_EQ(done.data().size(), 0u);
    std::remove(path);
}

#ifdef __unix__
static harmonics::HTensor make_tensor(float a, float b) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(DatasetProducerTest, HttpProducerFetchesRecords) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char c;
        std::string req;
        while (::recv(client, &c, 1, 0) > 0) {
            req.push_back(c);
            if (req.size() >= 4 && req.compare(req.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::ostringstream payload;
        harmonics::write_tensor(payload, make_tensor(1.f, 2.f));
        harmonics::write_tensor(payload, make_tensor(3.f, 4.f));
        std::string body = payload.str();
        std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        ::send(client, resp.c_str(), resp.size(), 0);
        ::send(client, body.data(), body.size(), 0);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::HttpProducer prod("127.0.0.1", port);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 1.f);
    EXPECT_EQ(db[1], 4.f);
    server.join();
}

TEST(DatasetProducerTest, AsyncHttpProducerFetchesRecords) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char c;
        std::string req;
        while (::recv(client, &c, 1, 0) > 0) {
            req.push_back(c);
            if (req.size() >= 4 && req.compare(req.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::ostringstream payload;
        harmonics::write_tensor(payload, make_tensor(7.f, 8.f));
        harmonics::write_tensor(payload, make_tensor(9.f, 10.f));
        std::string body = payload.str();
        std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        ::send(client, resp.c_str(), resp.size(), 0);
        ::send(client, body.data(), body.size(), 0);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::AsyncHttpProducer prod("127.0.0.1", port);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 7.f);
    EXPECT_EQ(db[0], 9.f);
    server.join();
}

TEST(DatasetProducerTest, HttpProducerCachesToDisk) {
    const char* cache = "http_cache.bin";
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char c;
        std::string req;
        while (::recv(client, &c, 1, 0) > 0) {
            req.push_back(c);
            if (req.size() >= 4 && req.compare(req.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::ostringstream payload;
        harmonics::write_tensor(payload, make_tensor(5.f, 6.f));
        std::string body = payload.str();
        std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        ::send(client, resp.c_str(), resp.size(), 0);
        ::send(client, body.data(), body.size(), 0);
        ::close(client);
        ::close(server_fd);
    });

    {
        harmonics::HttpProducer prod("127.0.0.1", port, "/", true, cache);
        EXPECT_EQ(prod.size(), 1u);
        auto t = prod.next();
        const float* d = reinterpret_cast<const float*>(t.data().data());
        EXPECT_EQ(d[1], 6.f);
    }

    server.join();

    harmonics::HttpProducer from_cache("127.0.0.1", port, "/", true, cache);
    EXPECT_EQ(from_cache.size(), 1u);
    auto r = from_cache.next();
    const float* dr = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(dr[0], 5.f);
    std::remove(cache);
}

TEST(DatasetProducerTest, HttpHdf5ProducerFetchesRecords) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char c;
        std::string req;
        while (::recv(client, &c, 1, 0) > 0) {
            req.push_back(c);
            if (req.size() >= 4 && req.compare(req.size() - 4, 4, "\r\n\r\n") == 0)
                break;
        }
        std::ostringstream payload;
        {
            harmonics::Hdf5Consumer writer(payload);
            writer.push(make_tensor(1.f, 2.f));
            writer.push(make_tensor(3.f, 4.f));
        }
        std::string body = payload.str();
        std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        ::send(client, resp.c_str(), resp.size(), 0);
        ::send(client, body.data(), body.size(), 0);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::HttpHdf5Producer prod("127.0.0.1", port);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(da[0], 1.f);
    EXPECT_EQ(db[0], 3.f);
    server.join();
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
