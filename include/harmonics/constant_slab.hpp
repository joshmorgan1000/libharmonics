#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace harmonics {

/** Maximum number of sensor or appendage slots. */
inline constexpr std::size_t MAX_VARIABLE_SLOTS = 4;

/** Maximum number of hidden neurons allocated for each slot. */
inline constexpr std::size_t MAX_SLOT_NEURONS = 96;

/**
 * @brief Fixed-size memory slab holding sensor and appendage slot data.
 *
 * Each slot reserves space for @ref MAX_SLOT_NEURONS values so the layout
 * is constant regardless of the active hidden size. Unused regions should be
 * masked out by the caller using the per-slot active flags.
 */
template <typename T> struct ConstantSlab {
    alignas(32) T sensor_data[MAX_VARIABLE_SLOTS * MAX_SLOT_NEURONS]{};
    alignas(32) T appendage_data[MAX_VARIABLE_SLOTS * MAX_SLOT_NEURONS]{};
    uint8_t sensor_active[MAX_VARIABLE_SLOTS]{};    ///< 1 when the slot is active
    uint8_t appendage_active[MAX_VARIABLE_SLOTS]{}; ///< 1 when the slot is active

    /** Clear all data and reset active flags to zero. */
    void clear() {
        std::memset(sensor_data, 0, sizeof(T) * MAX_VARIABLE_SLOTS * MAX_SLOT_NEURONS);
        std::memset(appendage_data, 0, sizeof(T) * MAX_VARIABLE_SLOTS * MAX_SLOT_NEURONS);
        std::memset(sensor_active, 0, MAX_VARIABLE_SLOTS);
        std::memset(appendage_active, 0, MAX_VARIABLE_SLOTS);
    }

    /** Return pointer to the start of the sensor slot @p idx. */
    T* sensor_slot(std::size_t idx) { return sensor_data + idx * MAX_SLOT_NEURONS; }
    const T* sensor_slot(std::size_t idx) const { return sensor_data + idx * MAX_SLOT_NEURONS; }

    /** Return pointer to the start of the appendage slot @p idx. */
    T* appendage_slot(std::size_t idx) { return appendage_data + idx * MAX_SLOT_NEURONS; }
    const T* appendage_slot(std::size_t idx) const {
        return appendage_data + idx * MAX_SLOT_NEURONS;
    }
};

} // namespace harmonics
