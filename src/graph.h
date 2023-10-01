/*
compute-graph-autodiff  Copyright (C) 2023 Enrico Guiraud
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c' for details.
*/
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"

/* A note on the use of shared_ptr<const T>

  They are not as bad as they might seem at first sight:
  - we need the pointer indirection anyways to break
    dependency cycles between types of graph nodes
  - the data shared is read-only because of the `const`
  - cannot use unique_ptr without duplicating large chunks
    of the graphs: the same node can participate to multiple
    operations, so shared ownership is actually a good fit
*/

namespace compute_graph_ad {

/// Inputs to a graph's eval function: a mapping from variable name to value.
using Inputs = absl::flat_hash_map<std::string, float>;

/// An operation in the compute graph (e.g. addition, multiplication).
// We need a virtual base class to break dependency cycles e.g. between
// Sum and Mul which they can point to each other.
class Op {
 public:
  virtual ~Op() {}

  /// Evaluate this operation on the inputs provided.
  virtual float eval(const Inputs& inputs) const noexcept = 0;
};

/// A sum operation, with two operands that can be operations themselves.
class Sum : public Op {
  std::shared_ptr<const Op> op1, op2;

 public:
  Sum(std::shared_ptr<const Op> op1, std::shared_ptr<const Op> op2)
      : op1(std::move(op1)), op2(std::move(op2)) {}

  float eval(const Inputs& inputs) const noexcept final;
};

/// A multiplication operation, with two operands that can be operations
/// themselves.
class Mul : public Op {
  std::shared_ptr<const Op> op1, op2;

 public:
  Mul(std::shared_ptr<const Op> op1, std::shared_ptr<const Op> op2)
      : op1(std::move(op1)), op2(std::move(op2)) {}

  float eval(const Inputs& inputs) const noexcept final;
};

/// A compute graph.
/// Can be combined with other graphs via operations like Sum and Mul,
/// and related math operators.
class Graph {
  std::shared_ptr<const Op> op;

 public:
  Graph(std::shared_ptr<const Op> op) : op(op) {}

  // using the hidden friend pattern not to pollute the global namespace:
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1601r0.pdf
  friend Graph operator+(const Graph& g1, const Graph& g2) {
    return Graph(std::make_shared<const Sum>(g1.op, g2.op));
  }

  friend Graph operator*(const Graph& g1, const Graph& g2) {
    return Graph(std::make_shared<const Mul>(g1.op, g2.op));
  }

  float eval(const Inputs& inputs) const noexcept;
};

/// A scalar constant.
class Const : public Op {
  float value;

 public:
  Const(float value) : value(value) {}

  float eval(const Inputs&) const noexcept final { return value; }

  /* operator+ */
  // using the hidden friend pattern to avoid polluting the global namespace:
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1601r0.pdf
  // TODO: find a less invasive way to define these operations.
  friend Graph operator+(const Const& c1, const Const& c2) {
    const auto g1 = Graph(std::make_shared<const Const>(c1));
    const auto g2 = Graph(std::make_shared<const Const>(c2));
    return g1 + g2;
  }

  friend Graph operator+(const Graph& g1, const Const& c2) {
    auto g2 = Graph{std::make_shared<const Const>(c2)};
    return g1 + g2;
  }

  friend Graph operator+(const Const& c1, const Graph& g2) { return g2 + c1; }

  /* operator* */
  friend Graph operator*(const Const& c1, const Const& c2) {
    const auto g1 = Graph(std::make_shared<const Const>(c1));
    const auto g2 = Graph(std::make_shared<const Const>(c2));
    return g1 * g2;
  }

  friend Graph operator*(const Graph& g1, const Const& c2) {
    auto g2 = Graph{std::make_shared<const Const>(c2)};
    return g1 * g2;
  }

  friend Graph operator*(const Const& c1, const Graph& g2) { return g2 * c1; }
};

/// A scalar variable: a named placeholder for an input to `evaluate`.
class Var : public Op {
  std::string name;

 public:
  Var(std::string_view name) : name(name) {}

  /* operator+ */
  // using the hidden friend pattern to avoid polluting the global namespace:
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1601r0.pdf
  friend Graph operator+(const Var& v1, const Var& v2) {
    const auto g1 = Graph(std::make_shared<const Var>(v1));
    const auto g2 = Graph(std::make_shared<const Var>(v2));
    return g1 + g2;
  }

  friend Graph operator+(const Const& c1, const Var& v2) {
    const auto g1 = Graph{std::make_shared<const Const>(c1)};
    const auto g2 = Graph{std::make_shared<const Var>(v2)};
    return g1 + g2;
  }

  friend Graph operator+(const Var& v1, const Const& c2) { return c2 + v1; }

  friend Graph operator+(const Graph& g1, const Var& v2) {
    auto g2 = Graph{std::make_shared<const Var>(v2)};
    return g1 + g2;
  }

  friend Graph operator+(const Var& v1, const Graph& g2) { return g2 + v1; }

  /* operator* */
  friend Graph operator*(const Var& v1, const Var& v2) {
    const auto g1 = Graph{std::make_shared<const Var>(v1)};
    const auto g2 = Graph{std::make_shared<const Var>(v2)};
    return g1 * g2;
  }

  friend Graph operator*(const Const& c1, const Var& v2) {
    const auto g1 = Graph{std::make_shared<const Const>(c1)};
    const auto g2 = Graph{std::make_shared<const Var>(v2)};
    return g1 * g2;
  }

  friend Graph operator*(const Var& v1, const Const& c2) { return c2 * v1; }

  friend Graph operator*(const Graph& g1, const Var& v2) {
    auto g2 = Graph{std::make_shared<const Var>(v2)};
    return g1 * g2;
  }

  friend Graph operator*(const Var& v1, const Graph& g2) { return g2 * v1; }

  float eval(const Inputs& inputs) const noexcept final;
};
}  // namespace compute_graph_ad