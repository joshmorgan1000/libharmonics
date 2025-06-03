#pragma once

#include "harmonics/int8_math.hpp"
#include <blake3.h>
#include <cstdint>
#include <cstring>

namespace harmonics {
extern "C" {

static uint8_t g_lineage_hash[32] = {};
static int8_t g_weights[36] = {};
static int8_t g_biases[6] = {};
static std::uint64_t g_step = 0;

inline void cb_forward(const int8_t* input, int8_t* action_out) {
    if (!input || !action_out)
        return;

    for (int o = 0; o < 6; ++o) {
        int16_t sum = static_cast<int16_t>(g_biases[o]);
        for (int i = 0; i < 6; ++i) {
            int16_t prod =
                static_cast<int16_t>(input[i]) * static_cast<int16_t>(g_weights[o * 6 + i]);
            sum += prod / 128;
        }
        if (sum > 127)
            sum = 127;
        else if (sum < -128)
            sum = -128;
        action_out[o] = static_cast<int8_t>(sum);
    }

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, action_out, 6);
    blake3_hasher_update(&h, g_lineage_hash, 32);
    blake3_hasher_finalize(&h, g_lineage_hash, 32);
}

inline void cb_train(const int8_t* input, const int8_t* target, uint8_t lr_shift,
                     uint8_t* energy_spent) {
    if (!input || !target || !energy_spent)
        return;

    uint8_t spent = 0;
    for (int o = 0; o < 6; ++o) {
        int16_t sum = static_cast<int16_t>(g_biases[o]);
        for (int i = 0; i < 6; ++i) {
            int16_t prod =
                static_cast<int16_t>(input[i]) * static_cast<int16_t>(g_weights[o * 6 + i]);
            sum += prod / 128;
        }
        int16_t diff = sum - static_cast<int16_t>(target[o]);
        spent += static_cast<uint8_t>(diff < 0 ? -diff : diff);

        int16_t upd = diff >> lr_shift;
        int16_t nb = static_cast<int16_t>(g_biases[o]) - upd;
        if (nb > 127)
            nb = 127;
        else if (nb < -128)
            nb = -128;
        g_biases[o] = static_cast<int8_t>(nb);

        for (int i = 0; i < 6; ++i) {
            int32_t grad = diff * static_cast<int16_t>(input[i]);
            int16_t wu = static_cast<int16_t>(grad >> (lr_shift + 7));
            int16_t nw = static_cast<int16_t>(g_weights[o * 6 + i]) - wu;
            if (nw > 127)
                nw = 127;
            else if (nw < -128)
                nw = -128;
            g_weights[o * 6 + i] = static_cast<int8_t>(nw);
        }
    }
    *energy_spent = spent;
    ++g_step;

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, input, 6);
    blake3_hasher_update(&h, target, 6);
    blake3_hasher_update(&h, g_weights, 36);
    blake3_hasher_update(&h, g_biases, 6);
    blake3_hasher_update(&h, &lr_shift, 1);
    blake3_hasher_update(&h, g_lineage_hash, 32);
    blake3_hasher_finalize(&h, g_lineage_hash, 32);
}

inline void cb_hash(uint8_t* out32) {
    if (!out32)
        return;

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, g_weights, 36);
    blake3_hasher_update(&h, g_biases, 6);
    blake3_hasher_update(&h, g_lineage_hash, 32);
    blake3_hasher_finalize(&h, out32, 32);
}

} // extern "C"
} // namespace harmonics
