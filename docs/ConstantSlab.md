# Constant Slab Memory Layout

The constant slab is a fixed size buffer used by the runtime to store sensor and
appendage activations. It guarantees a deterministic layout so kernels can work
without dynamic allocations.

## Slot Arrangement

Each slab exposes `MAX_VARIABLE_SLOTS` sensor and appendage slots. Every slot
reserves space for `MAX_SLOT_NEURONS` values even if only a subset are active.
Inactive regions are ignored by kernels using the per-slot `active` flags.

```
ConstantSlab<float> slab;
// 4 sensor slots and 4 appendage slots
// sensor_data and appendage_data are laid out contiguously in memory:
// [slot0 96 floats][slot1 96 floats][slot2 96 floats][slot3 96 floats]
```

Slots are 32-byte aligned so vectorised operations can load contiguous blocks
without extra indirection.

## Usage

The slab is cleared with `clear()` which zeros all values and resets the active
flags. Access helpers return pointers to the start of each slot:

```cpp
slab.clear();
float* s0 = slab.sensor_slot(0);
float* a1 = slab.appendage_slot(1);
```

Caller code is responsible for masking out inactive neurons when reading or
writing values. The runtime updates `sensor_active[i]` and
`appendage_active[i]` to indicate which slots contain valid data at a given
step.


