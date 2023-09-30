:construction: this project is a work in progress :construction:

# C++ compute graph autodiff

A proof of concept implementation of C++ compute graph autodifferentiation.

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
bazel run @hedron_compile_commands//:refresh_all
```
