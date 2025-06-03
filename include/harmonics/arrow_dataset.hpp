#pragma once

#include "harmonics/core.hpp"

#ifdef HARMONICS_HAS_ARROW
#include <arrow/c/abi.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#endif

namespace harmonics {

#ifdef HARMONICS_HAS_ARROW

/** Convert a simple Arrow column of float32 values into an HTensor. */
inline HTensor arrow_column_to_tensor(const ArrowSchema* schema, const ArrowArray* array) {
    if (!schema || !array || !schema->format)
        return {};
    // Only handle float32 columns for now
    if (std::strcmp(schema->format, "f") != 0)
        throw std::runtime_error("unsupported Arrow format");
    if (array->n_buffers < 2 || !array->buffers[1])
        throw std::runtime_error("invalid Arrow buffers");
    const float* data = reinterpret_cast<const float*>(array->buffers[1]) + array->offset;
    HTensor t{HTensor::DType::Float32, {static_cast<std::size_t>(array->length)}};
    t.data().resize(sizeof(float) * array->length);
    std::memcpy(t.data().data(), data, sizeof(float) * array->length);
    return t;
}

/** Producer yielding tensors from an Arrow struct array or record batch. */
class ArrowArrayProducer : public Producer {
  public:
    ArrowArrayProducer(const ArrowSchema* schema, const ArrowArray* array) {
        if (!schema || !array)
            return;
        // Treat children as columns forming each record
        std::size_t rows = static_cast<std::size_t>(array->length);
        records_.reserve(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            std::vector<float> values;
            for (int c = 0; c < schema->n_children; ++c) {
                const ArrowSchema* cs = schema->children[c];
                const ArrowArray* ca = array->children[c];
                if (!cs || !ca)
                    continue;
                if (std::strcmp(cs->format, "f") != 0)
                    throw std::runtime_error("only float32 columns supported");
                const float* col_data = reinterpret_cast<const float*>(ca->buffers[1]) + ca->offset;
                values.push_back(col_data[r]);
            }
            HTensor t{HTensor::DType::Float32, {values.size()}};
            t.data().resize(sizeof(float) * values.size());
            std::memcpy(t.data().data(), values.data(), sizeof(float) * values.size());
            records_.push_back(std::move(t));
        }
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        return records_[index_++ % records_.size()];
    }
    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};
};

#endif // HARMONICS_HAS_ARROW

} // namespace harmonics
