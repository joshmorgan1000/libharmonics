#pragma once

#include <blake3.h>
#include <cstdint>
#include <cstring>

namespace harmonics {
extern "C" {

static uint8_t g_lineage_hash[32] = {};

inline void cb_forward(const int8_t* input, int8_t* action_out) {
    if (!input || !action_out)
        return;
    for (int i = 0; i < 6; ++i)
        action_out[i] = input[i];

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, input, 6);
    blake3_hasher_update(&h, g_lineage_hash, 32);
    blake3_hasher_finalize(&h, g_lineage_hash, 32);
}

inline void cb_train(const int8_t* input, const int8_t* target, uint8_t lr_shift,
                     uint8_t* energy_spent) {
    (void)lr_shift;
    if (!input || !target || !energy_spent)
        return;
    uint8_t spent = 0;
    for (int i = 0; i < 6; ++i) {
        int16_t diff = static_cast<int16_t>(input[i]) - static_cast<int16_t>(target[i]);
        spent += static_cast<uint8_t>(diff < 0 ? -diff : diff);
    }
    *energy_spent = spent;

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, input, 6);
    blake3_hasher_update(&h, target, 6);
    blake3_hasher_update(&h, &lr_shift, 1);
    blake3_hasher_update(&h, g_lineage_hash, 32);
    blake3_hasher_finalize(&h, g_lineage_hash, 32);
}

inline void cb_hash(uint8_t* out32) {
    if (!out32)
        return;
    std::memcpy(out32, g_lineage_hash, 32);
}

} // extern "C"
} // namespace harmonics
