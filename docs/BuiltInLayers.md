# Built-in Layer Guide

This document describes the additional layers available in Harmonics: cross-attention, pooling and dropout. These layers are registered when `register_builtin_layers()` is called.

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

## Pooling Layers

Two pooling layers are provided for 1D float tensors: `max_pool` and `avg_pool`. Their window size is configured through `set_pool_window()`.

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

`dropout` zeros elements of a tensor with the given probability. The default rate is `0.5`. To use a different rate register the layer explicitly:

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
