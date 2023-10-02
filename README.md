[![ci](https://github.com/eguiraud/cpp-graph-autodiff/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eguiraud/cpp-graph-autodiff/actions/workflows/ci.yml)

# C++ compute graph autodiff

A **proof of concept** implementation of C++ compute graph autodifferentiation.

The library builds compute graphs made of multiplications and additions of variables and constants.

It can then evaluate the graph and its gradient w.r.t. the input variables at a point.

[Forward mode autodifferentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation) is used to compute the gradients.
Differently from most implementations, we return derivatives w.r.t. all input variables even if using forward mode.

## How does this look?

```cpp
// create variables and constants
const Var x{"x"};
const Var y{"y"};
const Var z{"z"};
const Const c{10.};

// build an expression (aka compute graph)
// currently only additions and multiplications are supported

// g(x,y,z) = yx^3 + xyz + cz(x + y) + c
// dg/dx = 3yx^2 + yz + cz
// dg/dy = x^3 + xz + cz
// dg/dz = xy + cx + cy
const Graph g = x*x*x*y + x*y*z + c*z*(x + y) + c;

// now given some concrete values for the input variables...
const Inputs inputs = {{"x", 2.}, {"y", 3.}, {"z", 4.}};

// ...we can evaluate the expression and its gradient
// `grads` contains (dg/dx, dg/dy, dg/dz): variables are in alphabetical order
const auto &[value, grads] = g.eval_grad(inputs);

// g(2,3,4) = 24 + 24 + 200 + 10 = 258
EXPECT_FLOAT_EQ(value, 258.);

// dg/dx(2,3,4) = 36 + 12 + 40 = 88
EXPECT_FLOAT_EQ(grads(0), 88.);
// dg/dy(2,3,4) = 8 + 8 + 40 = 56
EXPECT_FLOAT_EQ(grads(1), 56.);
// dg/dz(2,3,4) = 6 + 20 + 30 = 56
EXPECT_FLOAT_EQ(grads(2), 56.);
```

## Run tests

```shell
bazel run '//tests:test'
```

## For developers: producing a compilation database

This project uses [bazel-compile-commands-extractor](https://github.com/hedronvision/bazel-compile-commands-extractor)
to generate a compilation database (i.e. a file called `compile_commands.json`) which tools like [clangd](https://clangd.llvm.org/),
or [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) can use to understand your source files.

To create or refresh the compilation database, run:

```
bazel build //tests:test # build everything and generate protobuf files
bazel run @hedron_compile_commands//:refresh_all # refresh compile_commands.json
```
