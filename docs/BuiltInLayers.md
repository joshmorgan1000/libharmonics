# Built-in Layer Guide

This document describes the additional layers available in Harmonics: cross-attention, multi-head self-attention, pooling and dropout. These layers are registered when `register_builtin_layers()` is called.

## Cross-Attention

`cross_attention` operates on 2D float tensors. For each row it computes a softmax over the columns and outputs the weighted sum replicated across the row. The temperature is controlled by `set_attention_temperature()`.

Example DSL usage:

```harmonics
layer query;
cycle {
    query -(cross_attention)-> query;
}
```

Programmatically the layer can be accessed via `getLayer("cross_attention")`.

## Multi-Head Self-Attention

`multihead_attention` operates on 1D float tensors. The number of heads is
configured via `set_attention_heads()` and the temperature via
`set_attention_temperature()`. Each head attends to a slice of the input and the
results are averaged.

Example DSL usage:

```harmonics
layer hidden;
cycle {
    hidden -(multihead_attention)-> hidden;
}
```

Use `getLayer("multihead_attention")` to access the layer programmatically.

## Pooling Layers

Two pooling layers are provided for 1D tensors: `max_pool` and `avg_pool`. They
operate on 32-bit floats and unsigned 8-bit integers. Their window size is
configured through `set_pool_window()`.

DSL example:

```harmonics
layer input;
layer pooled;
cycle {
    input -(max_pool)-> pooled;
}
```

Retrieve them at runtime with `getLayer("max_pool")` or `getLayer("avg_pool")`.

## Dropout

`dropout` zeros elements of a tensor with the given probability. It works with all
numeric tensor types. The default rate is `0.5`. To use a different rate register
the layer explicitly:

```cpp
registerLayer("dropout", std::make_shared<harmonics::DropoutLayer>(0.2f));
```

In a graph:

```harmonics
layer hidden;
cycle {
    hidden -(dropout)-> hidden;
}
```

When CUDA support is enabled (`HARMONICS_ENABLE_CUDA=1`) dropout on `float32`
tensors is executed by a dedicated device kernel.
