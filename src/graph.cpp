/*
compute-graph-autodiff  Copyright (C) 2023 Enrico Guiraud
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c' for details.
*/
#include "graph.h"

#include <cassert>

using namespace compute_graph_ad;

float Sum::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) + op2->eval(inputs);
}

float Mul::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) * op2->eval(inputs);
}

float Graph::eval(const Inputs& inputs) const noexcept {
  assert(op);
  return op->eval(inputs);
}

float Var::eval(const Inputs& inputs) const noexcept {
  auto var_it = inputs.find(name);
  if (var_it == inputs.end()) {
    std::abort();  // TODO also log an error
  }
  return var_it->second;
}
