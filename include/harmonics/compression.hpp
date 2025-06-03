#pragma once

#include "harmonics/core.hpp"
#include "harmonics/serialization.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <zstd.h>

namespace harmonics {

inline HTensor compress_tensor(const HTensor& t) {
    std::ostringstream out;
    write_tensor(out, t);
    auto str = out.str();
    size_t bound = ZSTD_compressBound(str.size());
    std::vector<std::byte> comp(bound);
    size_t c = ZSTD_compress(comp.data(), bound, str.data(), str.size(), /*level=*/1);
    if (ZSTD_isError(c))
        throw std::runtime_error("tensor compression failed");
    comp.resize(c);
    return HTensor{HTensor::DType::UInt8, {comp.size()}, std::move(comp)};
}

inline HTensor decompress_tensor(const HTensor& t) {
    if (t.dtype() != HTensor::DType::UInt8)
        return t;
    const auto& bytes = t.data();
    size_t decomp_size = ZSTD_getFrameContentSize(bytes.data(), bytes.size());
    std::vector<std::byte> out(decomp_size);
    size_t got = ZSTD_decompress(out.data(), decomp_size, bytes.data(), bytes.size());
    if (ZSTD_isError(got))
        throw std::runtime_error("tensor decompression failed");
    std::string tmp(reinterpret_cast<const char*>(out.data()), got);
    std::istringstream in(tmp);
    return read_tensor(in);
}

class ZstdProducer : public Producer {
  public:
    explicit ZstdProducer(std::unique_ptr<Producer> inner) : inner_{std::move(inner)} {}

    HTensor next() override { return decompress_tensor(inner_->next()); }
    std::size_t size() const override { return inner_->size(); }

  private:
    std::unique_ptr<Producer> inner_{};
};

class ZstdConsumer : public Consumer {
  public:
    explicit ZstdConsumer(std::unique_ptr<Consumer> inner) : inner_{std::move(inner)} {}

    void push(const HTensor& t) override { inner_->push(compress_tensor(t)); }

  private:
    std::unique_ptr<Consumer> inner_{};
};

} // namespace harmonics
