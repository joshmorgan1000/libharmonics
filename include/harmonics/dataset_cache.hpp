#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "harmonics/dataset.hpp"

namespace harmonics {

/**
 * @brief Wrap a producer and cache its records to disk.
 *
 * If the cache file already exists it is loaded via @ref Hdf5Producer.
 * Otherwise samples are read from the provided producer and written to
 * the cache for reuse on subsequent runs.
 */
class CachedProducer : public Producer {
  public:
    CachedProducer(std::shared_ptr<Producer> inner, const std::string& path, bool compress = false,
                   bool validate = false)
        : path_{path}, compress_{compress}, validate_{validate} {
        std::ifstream chk(path_, std::ios::binary);
        if (chk) {
            prod_ = std::make_shared<Hdf5Producer>(path_, validate_);
        } else {
            inner_ = std::move(inner);
            if (inner_)
                consumer_ = std::make_shared<Hdf5Consumer>(path_, compress_);
        }
    }

    HTensor next() override {
        if (prod_)
            return prod_->next();
        if (!inner_)
            return {};
        auto t = inner_->next();
        if (consumer_)
            consumer_->push(t);
        ++index_;
        if (index_ >= inner_->size()) {
            consumer_.reset();
            prod_ = std::make_shared<Hdf5Producer>(path_, validate_);
            inner_.reset();
            index_ = 0;
        }
        return t;
    }

    std::size_t size() const override {
        if (prod_)
            return prod_->size();
        return inner_ ? inner_->size() : 0;
    }

    /// Download records from a remote source into the local cache.
    ///
    /// If the cache already exists the download is resumed by skipping
    /// any records that are present locally and appending the missing
    /// tensors. This mirrors the behaviour of the dataset_cache_cli tool
    /// which allows interrupted transfers to continue without starting
    /// over.
    void download(std::shared_ptr<Producer> remote) {
        if (!remote)
            return;

        std::size_t existing = 0;
        {
            std::ifstream chk(path_, std::ios::binary);
            if (chk) {
                Hdf5Producer tmp(path_, validate_);
                existing = tmp.size();
            }
        }

        CheckpointHdf5Consumer writer(path_, compress_);
        for (std::size_t i = 0; i < existing; ++i) {
            HTensor skip = remote->next();
            if (skip.data().empty()) {
                prod_ = std::make_shared<Hdf5Producer>(path_, validate_);
                return;
            }
        }

        while (true) {
            HTensor t = remote->next();
            if (t.data().empty())
                break;
            writer.push(t);
        }
        prod_ = std::make_shared<Hdf5Producer>(path_, validate_);
    }

    /// Upload the cached records to a remote consumer.
    void upload(std::shared_ptr<Consumer> remote) const {
        if (!remote)
            return;
        std::ifstream chk(path_, std::ios::binary);
        if (!chk)
            return;
        Hdf5Producer reader(path_, validate_);
        for (std::size_t i = 0; i < reader.size(); ++i)
            remote->push(reader.next());
        remote->push({});
    }

  private:
    std::string path_{};
    bool compress_{false};
    bool validate_{false};
    std::shared_ptr<Producer> inner_{};
    std::shared_ptr<Hdf5Consumer> consumer_{};
    std::shared_ptr<Hdf5Producer> prod_{};
    std::size_t index_{0};
};

/**
 * @brief Distributed cache that syncs records over the network.
 *
 * When the local cache file is missing it will be downloaded from a
 * remote producer. On destruction any cached records are uploaded to a
 * remote consumer so other machines can fetch them later.
 */
class DistributedCachedProducer : public Producer {
  public:
    DistributedCachedProducer(std::shared_ptr<Producer> inner, const std::string& path,
                              std::shared_ptr<Producer> remote_src = nullptr,
                              std::shared_ptr<Consumer> remote_dst = nullptr, bool compress = false,
                              bool validate = false)
        : cache_{std::move(inner), path, compress, validate}, remote_src_{std::move(remote_src)},
          remote_dst_{std::move(remote_dst)} {
        std::ifstream chk(path, std::ios::binary);
        if (!chk && remote_src_)
            cache_.download(remote_src_);
    }

    ~DistributedCachedProducer() {
        if (remote_dst_)
            cache_.upload(remote_dst_);
    }

    HTensor next() override { return cache_.next(); }
    std::size_t size() const override { return cache_.size(); }

  private:
    CachedProducer cache_;
    std::shared_ptr<Producer> remote_src_{};
    std::shared_ptr<Consumer> remote_dst_{};
};

} // namespace harmonics
