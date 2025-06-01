#pragma once

#include "harmonics/core.hpp"
#include "harmonics/serialization.hpp"

#include <memory>
#include <sstream>
#include <zstd.h>

namespace harmonics {

inline HTensor compress_tensor(const HTensor& t) {
    std::ostringstream out;
    write_tensor(out, t);
    auto str = out.str();
    size_t bound = ZSTD_compressBound(str.size());
    std::string comp(bound, '\0');
    size_t c = ZSTD_compress(comp.data(), bound, str.data(), str.size(), 1);
    if (ZSTD_isError(c))
        throw std::runtime_error("tensor compression failed");
    comp.resize(c);
    std::vector<std::byte> data(comp.begin(), comp.end());
    return HTensor{HTensor::DType::UInt8, {data.size()}, std::move(data)};
}

inline HTensor decompress_tensor(const HTensor& t) {
    if (t.dtype() != HTensor::DType::UInt8)
        return t;
    std::string comp(reinterpret_cast<const char*>(t.data().data()), t.data().size());
    size_t decomp_size = ZSTD_getFrameContentSize(comp.data(), comp.size());
    std::string out(decomp_size, '\0');
    size_t got = ZSTD_decompress(out.data(), decomp_size, comp.data(), comp.size());
    if (ZSTD_isError(got))
        throw std::runtime_error("tensor decompression failed");
    std::istringstream in(out);
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
