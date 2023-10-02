#include <gtest/gtest.h>

#include "absl/status/statusor.h"
#include "src/graph.h"

using namespace compute_graph_ad;

TEST(Graph, SumEval) {
  const Const c(2.);
  const Var x("x");
  const Inputs inputs{{"x", 3.}};

  const Graph g1 = x + x;  // Var + Var
  EXPECT_FLOAT_EQ(g1.eval(inputs), 6.);

  const Graph g2 = c + c;  // Const + Const
  EXPECT_FLOAT_EQ(g2.eval(inputs), 4.);

  const Graph g3 = x + c;  // Var + Const
  const Graph g4 = c + x;  // Const + Var
  for (const auto &g : {g3, g4}) EXPECT_FLOAT_EQ(g.eval(inputs), 5.);

  const Graph g5 = x + g1;  // Var + Graph
  const Graph g6 = g1 + x;  // Graph + Var
  for (const auto &g : {g5, g6}) EXPECT_FLOAT_EQ(g.eval(inputs), 9.);

  const Graph g7 = c + g1;  // Const + Graph
  const Graph g8 = g1 + c;  // Graph + Const
  for (const auto &g : {g7, g8}) EXPECT_FLOAT_EQ(g.eval(inputs), 8.);

  // Graph and Graph
  const auto g9 = g1 + g1;  // Graph + Graph
  EXPECT_FLOAT_EQ(g9.eval(inputs), 12.);
}

TEST(Graph, MulEval) {
  const Const c(2.);
  const Var x("x");
  const Inputs inputs{{"x", 3.}};

  const Graph g1 = x * x;  // Var * Var
  EXPECT_FLOAT_EQ(g1.eval(inputs), 9.);

  const Graph g2 = c * c;  // Const * Const
  EXPECT_FLOAT_EQ(g2.eval(inputs), 4.);

  const Graph g3 = x * c;  // Var * Const
  const Graph g4 = c * x;  // Const * Var
  for (const auto &g : {g3, g4}) EXPECT_FLOAT_EQ(g.eval(inputs), 6.);

  const Graph g5 = x * g1;  // Var * Graph
  const Graph g6 = g1 * x;  // Graph * Var
  for (const auto &g : {g5, g6}) EXPECT_FLOAT_EQ(g.eval(inputs), 27.);

  const Graph g7 = c * g1;  // Const * Graph
  const Graph g8 = g1 * c;  // Graph * Const
  for (const auto &g : {g7, g8}) EXPECT_FLOAT_EQ(g.eval(inputs), 18.);

  // Graph and Graph
  const auto g9 = g1 * g1;  // Graph * Graph
  EXPECT_FLOAT_EQ(g9.eval(inputs), 81.);
}

TEST(Tests, WriteAndReadGraph) {
  const Const c{20.};
  const Var x{"x"};
  const Inputs inputs{{"x", 2.}};

  const Graph g1 = x + c * x;

  // write out
  const absl::Status status = to_file(g1, "test.pb");
  ASSERT_TRUE(status.ok());

  // read in
  const absl::StatusOr<Graph> gs = from_file("test.pb");
  ASSERT_TRUE(gs.ok());

  // check the graph still evaluates correctly
  const Graph &g2 = gs.value();
  EXPECT_FLOAT_EQ(g2.eval(inputs), 42.);
}

TEST(Tests, SumGradient) {
  const Var x{"x"};
  const Const c{2.};
  const Graph g = x + c;
  const auto &[value, grads] = g.eval_grad({{"x", 3.}});
  EXPECT_FLOAT_EQ(value, 5.);
  EXPECT_EQ(grads.size(), 1);
  EXPECT_FLOAT_EQ(grads(0), 1.);
}

TEST(Tests, MulGradient) {
  const Var x{"x"};
  const Const c{2.};
  const Graph g = x * c;
  const auto &[value, grads] = g.eval_grad({{"x", 3.}});
  EXPECT_FLOAT_EQ(value, 6.);
  EXPECT_EQ(grads.size(), 1);
  EXPECT_FLOAT_EQ(grads(0), 2.);
}

TEST(Tests, MultipleVarGradient) {
  const Var x{"x"};
  const Var y{"y"};
  const Var z{"z"};
  const Const c{10.};

  // clang-format off
  // g(x,y,z) = yx^3 + xyz + cz(x + y) + c
  // dg/dx = 3yx^2 + yz + cz
  // dg/dy = x^3 + xz + cz
  // dg/dz = xy + cx + cy
  const Graph g = x*x*x*y + x*y*z + c*z*(x + y) + c;
  // clang-format on

  const Inputs inputs = {{"x", 2.}, {"y", 3.}, {"z", 4.}};
  const auto &[value, grads] = g.eval_grad(inputs);

  // g(2,3,4) = 24 + 24 + 200 + 10 = 258
  EXPECT_FLOAT_EQ(value, 258.);

  // dg/dx(2,3,4) = 36 + 12 + 40 = 88
  EXPECT_FLOAT_EQ(grads(0), 88.);
  // dg/dy(2,3,4) = 8 + 8 + 40 = 56
  EXPECT_FLOAT_EQ(grads(1), 56.);
  // dg/dz(2,3,4) = 6 + 20 + 30 = 56
  EXPECT_FLOAT_EQ(grads(2), 56.);
}
