[![ci](https://github.com/eguiraud/cpp-graph-autodiff/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eguiraud/cpp-graph-autodiff/actions/workflows/ci.yml)

# C++ compute graph autodiff

A **proof of concept** implementation of C++ compute graph autodifferentiation.

The library builds compute graphs made of multiplications and additions of variables and constants.

It can then evaluate the graph and its gradient w.r.t. the input variables at a point.

[Forward mode autodifferentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation) is used to compute the gradients.
Differently from most implementations, we return derivatives with respect to all input variables, calculated in a single pass, even if using forward mode.

## How does this look?

```cpp
#include "graph_autodiff/graph.h"
using namespace graph_autodiff;

// create variables and constants
const Var x{"x"};
const Var y{"y"};
const Var z{"z"};
const Const c{10.};

// build an expression (aka compute graph)
// g(x,y,z) = yx^3 + xyz + cz(x + y) + c
const Graph g = x*x*x*y + x*y*z + c*z*(x + y) + c;

// now given some concrete values for the input variables...
const Inputs inputs = {{"x", 2.}, {"y", 3.}, {"z", 4.}};

// ...we can evaluate the expression and its gradient
// `grads` contains (dg/dx, dg/dy, dg/dz): variables are in alphabetical order
const auto &[value, grads] = g.eval_grad(inputs);

// g(2,3,4) = 24 + 24 + 200 + 10 = 258
EXPECT_FLOAT_EQ(value, 258.);

// dg/dx(2,3,4) = 3yx^2 + yz + cz = 36 + 12 + 40 = 88
EXPECT_FLOAT_EQ(grads(0), 88.);

// dg/dy(2,3,4) = x^3 + xz + cz = 8 + 8 + 40 = 56
EXPECT_FLOAT_EQ(grads(1), 56.);

// dg/dz(2,3,4) = xy + cx + cy = 6 + 20 + 30 = 56
EXPECT_FLOAT_EQ(grads(2), 56.);
```

### Ser/Deserialization of compute graphs

`Graph` objects are written to and read from files via [protobuf](https://protobuf.dev).
For example, continuing from above:

```cpp
// write out
const absl::Status status = to_file(g, "mygraph.pb");
ASSERT_TRUE(status.ok());

// read in
const absl::StatusOr<Graph> gs = from_file("mygraph.pb");
ASSERT_TRUE(gs.ok());
```

## Using this library

`cpp-graph-autodiff` can be imported into your project as a [Bazel module](https://bazel.build/external/module).
Just add this line to your `MODULE.bazel` file:
```
bazel_dep(name = "cpp-graph-autodiff", version = "0.1-alpha")
```
and then tell Bazel to use this project's dedicated registry, e.g.:

```shell
bazel build --enable_bzlmod \
	--registry https://bcr.bazel.build \
	--registry https://raw.githubusercontent.com/eguiraud/cpp-graph-autodiff/bazel-registry/bazel-registry \
	--cxxopt='-std=c++17' \
	'//...'
```
You can also add these options to your `.bazelrc`, of course. 

## For developers

### Running tests

```shell
bazel test --test_output=all '//...'
```

### Producing a compilation database

This project uses [bazel-compile-commands-extractor](https://github.com/hedronvision/bazel-compile-commands-extractor)
to generate a compilation database: a file called `compile_commands.json` which tools like [clangd](https://clangd.llvm.org/),
or [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) can use to better interpret the project's source files.

To create or refresh the compilation database, run:

```shell
bazel build '//...' # build everything and generate protobuf files
bazel run @hedron_compile_commands//:refresh_all # refresh compile_commands.json
```
