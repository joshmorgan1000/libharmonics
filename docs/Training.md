# Training API

This document describes the training facilities provided by Harmonics.

Training can occur on a single machine or across multiple nodes when a graph is partitioned by layer.  The runtime allows such partitions by treating the boundary tensors as producers and consumers; the user supplies the message transport.

## FitOptions

`FitOptions` controls how the parameters of a graph are updated during
training.

- `learning_rate` – step size for weight updates (default `0.01`).
- `optimizer` – algorithm used to apply gradients. Available optimizers:
  - `Optimizer::SGD` – stochastic gradient descent.
  - `Optimizer::Adam` – adaptive moment estimation.
  - `Optimizer::RMSProp` – RMSProp variant.
  - `Optimizer::AdamW` – Adam with decoupled weight decay.
  - `Optimizer::LAMB` – Layer-wise Adaptive Moments optimiser.

## Running training

Use `HarmonicGraph::fit` to execute the training cycle for a fixed
number of epochs or a duration. The call returns the final
`CycleState` containing learned weights.

```cpp
harmonics::FitOptions opt;
opt.learning_rate = 0.001f;
opt.optimizer = harmonics::Optimizer::Adam;

auto final = g.fit(5, harmonics::make_auto_policy(), opt);
```

For arbitrary stopping conditions, `fit_until` accepts a predicate that
is evaluated after each forward pass.

## Advanced options

`FitOptions` exposes several additional fields that tune the training loop:

- `grad_clip` &ndash; if greater than zero, gradient tensors are clamped to
  `[-grad_clip, grad_clip]` before updates are applied.
- `early_stop_patience` &ndash; when non&ndash;zero, training stops after this many
  consecutive steps without an improvement in gradient norm.
- `early_stop_delta` &ndash; minimum change in gradient norm required to reset the
  early stop counter.
- `weight_decay` &ndash; decay factor used by AdamW and LAMB (default `0`).
- `accumulate_steps` &ndash; number of forward passes to gather gradients before
  updating parameters (default `1`).
- `lr_schedule_fp` &ndash; optional callback returning the learning rate for
  each step.

Example configuration with Adam and gradient clipping:

```cpp
harmonics::FitOptions opt;
opt.learning_rate = 0.001f;
opt.optimizer = harmonics::Optimizer::Adam;
opt.grad_clip = 1.0f;
opt.early_stop_patience = 3;
opt.early_stop_delta = 0.01f;
auto state = g.fit(100, harmonics::make_auto_policy(), opt);
```

To decay the learning rate every ten steps:

```cpp
harmonics::ExponentialDecaySchedule sched{0.001f, 0.9f, 10};
opt.lr_schedule_fp = sched;
```

## Training until a custom condition

Instead of passing a fixed epoch count, `fit_until` runs until a predicate
returns `true`. The predicate receives the current `CycleState`:

```cpp
auto done = [&](const harmonics::CycleState&) {
    return some_condition();
};
auto state = g.fit_until(done, harmonics::make_auto_policy(), opt);
```

## Fixed precision

Use a `PrecisionPolicy` to force all activations to a specific bit width. The
policy applies equally to training and inference. For example, to run with
8‑bit activations:

`make_auto_policy()` uses the available hardware to pick a suitable
mixed‑precision format automatically. On GPUs this typically means 16‑bit
activations while gradients accumulate in 32‑bit.

```cpp
auto bits8 = harmonics::make_max_bits_policy(8);
auto final = g.fit(5, bits8, opt);
auto out = g.inference({}, bits8);

// lower precisions are available for aggressive quantisation
auto bits4 = harmonics::make_max_bits_policy(4);
auto bits3 = harmonics::make_max_bits_policy(3);
```

Precisions below eight bits use dedicated kernels for core operations. The
3‑bit matrix multiplication shader `int3_matmul` is selected automatically when
`make_max_bits_policy(3)` is active so experiments with extreme quantisation
require no additional changes.

`make_hardware_policy()` is used by default when no policy is supplied.

## Saving graphs and weights

The final `CycleState` contains the learned weights for each layer. Use the
serialization helpers to persist a trained graph:

```cpp
#include <harmonics/serialization.hpp>

std::ofstream gfile("model.hgr", std::ios::binary);
save_graph(g, gfile);

std::ofstream wfile("weights.hnwt", std::ios::binary);
save_weights(state.weights, wfile);
```

Later the files can be loaded with `load_graph` and `load_weights` to recreate
the runtime state.

## WebSocket training visualizer

`WebSocketTrainingVisualizer` streams progress metrics to another process over a WebSocket connection. Assign an instance to `FitOptions::progress` to emit a four element tensor containing the step index, gradient L2 norm, loss value and learning rate.

```cpp
#include <harmonics/training_visualizer.hpp>

harmonics::FitOptions opt;
opt.progress = harmonics::WebSocketTrainingVisualizer("127.0.0.1", 8080, "/metrics");

auto final = g.fit(5, harmonics::make_auto_policy(), opt);
```

A simple WebSocket server can collect these values and plot them live.

## Browser dashboard

A small Node.js application under `examples/training_dashboard` renders the streamed metrics in a browser. Start the server after installing the `ws` package:

```bash
cd examples/training_dashboard
npm install ws
node server.js
```

Open `http://localhost:8080` to view the interactive chart. The dashboard includes pause/resume controls and shows a 10 step moving average of the gradient norm alongside the loss value and learning rate. The screenshot `docs/assets/training_dashboard.png` illustrates the default layout.

## Distributed parameter server example

A minimal parameter server is provided under
`examples/distributed_parameter_server_example.cpp`. The server publishes the
current parameters over one gRPC stream while a worker sends gradients back on a
second stream. After applying the gradient, the updated parameter is broadcast
to the worker. This demonstrates how Harmonics I/O primitives can coordinate a
distributed training loop. See the runtime guide for build and run instructions.

