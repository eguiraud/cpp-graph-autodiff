#include <gtest/gtest.h>

#include "src/graph.h"

using namespace compute_graph_ad;

TEST(Graph, SumEval) {
  const Const c(2.);
  const Var x("x");
  const Inputs inputs{{"x", 3.}};

  const Graph g1 = x + x;  // Var + Var
  EXPECT_DOUBLE_EQ(g1.eval(inputs), 6.);

  const Graph g2 = c + c;  // Const + Const
  EXPECT_DOUBLE_EQ(g2.eval(inputs), 4.);

  const Graph g3 = x + c;  // Var + Const
  const Graph g4 = c + x;  // Const + Var
  for (const auto &g: {g3, g4})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 5.);

  const Graph g5 = x + g1;  // Var + Graph
  const Graph g6 = g1 + x;  // Graph + Var
  for (const auto &g: {g5, g6})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 9.);

  const Graph g7 = c + g1;  // Const + Graph
  const Graph g8 = g1 + c;  // Graph + Const
  for (const auto &g: {g7, g8})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 8.);

  // Graph and Graph
  const auto g9 = g1 + g1;  // Graph + Graph
  EXPECT_DOUBLE_EQ(g9.eval(inputs), 12.);
}

TEST(Graph, MulEval) {
  const Const c(2.);
  const Var x("x");
  const Inputs inputs{{"x", 3.}};

  const Graph g1 = x * x;  // Var * Var
  EXPECT_DOUBLE_EQ(g1.eval(inputs), 9.);

  const Graph g2 = c * c;  // Const * Const
  EXPECT_DOUBLE_EQ(g2.eval(inputs), 4.);

  const Graph g3 = x * c;  // Var * Const
  const Graph g4 = c * x;  // Const * Var
  for (const auto &g: {g3, g4})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 6.);

  const Graph g5 = x * g1;  // Var * Graph
  const Graph g6 = g1 * x;  // Graph * Var
  for (const auto &g: {g5, g6})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 27.);

  const Graph g7 = c * g1;  // Const * Graph
  const Graph g8 = g1 * c;  // Graph * Const
  for (const auto &g: {g7, g8})
    EXPECT_DOUBLE_EQ(g.eval(inputs), 18.);

  // Graph and Graph
  const auto g9 = g1 * g1;  // Graph * Graph
  EXPECT_DOUBLE_EQ(g9.eval(inputs), 81.);
}