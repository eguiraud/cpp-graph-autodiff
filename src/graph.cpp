/*
cpp-graph-autodiff  Copyright (C) 2023 Enrico Guiraud
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions: see LICENSE.
*/
#include "graph.h"

#include <cassert>
#include <cstddef>  // std::size_t
#include <fstream>
#include <iterator>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fmt/core.h"
#include "src/graph.pb.h"

using namespace compute_graph_ad;
namespace gpb = graph_proto;

namespace {
std::unique_ptr<const Op> op_from_proto(const gpb::Graph& gproto) {
  std::unique_ptr<const Op> op;

  switch (gproto.Op_case()) {
    case gpb::Graph::OpCase::kSum:
      op = Sum::from_proto(gproto.sum());
      break;
    case gpb::Graph::OpCase::kMul:
      op = Mul::from_proto(gproto.mul());
      break;
    case gpb::Graph::OpCase::kVar:
      op = Var::from_proto(gproto.var());
      break;
    case gpb::Graph::OpCase::kConst:
      op = Const::from_proto(gproto.const_());
      break;
    case gpb::Graph::OpCase::OP_NOT_SET:
      std::abort();  // TODO and log: this should never happen
      break;
  }

  return op;
}

std::size_t find_var_idx(std::string_view name, const Inputs& inputs) {
  // TODO this should _not_ be in the hot loop. do it once at the beginning.
  std::vector<std::string> var_names;
  var_names.reserve(inputs.size());
  absl::c_transform(inputs, std::back_inserter(var_names),
                    [](const auto& p) { return p.first; });
  absl::c_sort(var_names);

  auto it = absl::c_find(var_names, name);
  return std::distance(var_names.begin(), it);
}
}  // end of anonymous namespace

float Sum::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) + op2->eval(inputs);
}

float Sum::eval_grad(const Inputs& inputs,
                     Eigen::Map<Eigen::RowVectorXf>& grad_out) const noexcept {
  // row-major so we can give out views over the different rows to callees
  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajorBit> jacobian(
      2, inputs.size());

  Eigen::Map<Eigen::RowVectorXf> first_row_view(jacobian.data(),
                                                jacobian.cols());
  const float value1 = op1->eval_grad(inputs, first_row_view);

  Eigen::Map<Eigen::RowVectorXf> second_row_view(
      jacobian.data() + jacobian.rowStride(), jacobian.cols());
  const float value2 = op2->eval_grad(inputs, second_row_view);

  // the vector in the vector-jacobian product is just Ones() for a Sum
  grad_out = Eigen::RowVector2f::Ones() * jacobian;

  return value1 + value2;
}

gpb::Graph Sum::to_proto() const noexcept {
  gpb::Sum sum;
  *sum.mutable_op1() = op1->to_proto();
  *sum.mutable_op2() = op2->to_proto();

  gpb::Graph ret;
  *ret.mutable_sum() = std::move(sum);

  return ret;
}

std::unique_ptr<Sum> Sum::from_proto(const gpb::Sum& sproto) noexcept {
  return std::make_unique<Sum>(op_from_proto(sproto.op1()),
                               op_from_proto(sproto.op2()));
}

float Mul::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) * op2->eval(inputs);
}

float Mul::eval_grad(const Inputs& inputs,
                     Eigen::Map<Eigen::RowVectorXf>& grad_out) const noexcept {
  // row-major so we can give out views over the different rows to callees
  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajorBit> jacobian(
      2, inputs.size());

  Eigen::Map<Eigen::RowVectorXf> first_row_view(jacobian.data(),
                                                jacobian.cols());
  const float value1 = op1->eval_grad(inputs, first_row_view);

  Eigen::Map<Eigen::RowVectorXf> second_row_view(
      jacobian.data() + jacobian.rowStride(), jacobian.cols());
  const float value2 = op2->eval_grad(inputs, second_row_view);

  // dMul/dvalue_i for value1*value2 is (value2, value1)
  Eigen::RowVector2f dmul_dops;
  dmul_dops << value2, value1;
  grad_out = dmul_dops * jacobian;

  return value1 * value2;
}

gpb::Graph Mul::to_proto() const noexcept {
  gpb::Mul mul;
  *mul.mutable_op1() = op1->to_proto();
  *mul.mutable_op2() = op2->to_proto();

  gpb::Graph ret;
  *ret.mutable_mul() = std::move(mul);

  return ret;
}

std::unique_ptr<Mul> Mul::from_proto(const gpb::Mul& mproto) noexcept {
  return std::make_unique<Mul>(op_from_proto(mproto.op1()),
                               op_from_proto(mproto.op2()));
}

float Graph::eval(const Inputs& inputs) const noexcept {
  assert(op);
  return op->eval(inputs);
}

std::pair<float, Eigen::RowVectorXf> Graph::eval_grad(
    const Inputs& inputs) const noexcept {
  assert(op);

  Eigen::RowVectorXf grads(inputs.size());
  Eigen::Map<Eigen::RowVectorXf> grads_view(grads.data(), grads.size());
  return {op->eval_grad(inputs, grads_view), grads};
}

float Const::eval_grad(
    const Inputs& inputs,
    Eigen::Map<Eigen::RowVectorXf>& grad_out) const noexcept {
  // derivatives of a constant are all zero
  for (std::size_t i = 0; i < inputs.size(); ++i) grad_out[i] = 0;
  return value;
}

gpb::Graph Const::to_proto() const noexcept {
  gpb::Const c;
  c.set_value(value);

  gpb::Graph ret;
  *ret.mutable_const_() = std::move(c);

  return ret;
}

std::unique_ptr<Const> Const::from_proto(const gpb::Const& cproto) noexcept {
  return std::make_unique<Const>(cproto.value());
}

float Var::eval(const Inputs& inputs) const noexcept {
  auto var_it = inputs.find(name);
  if (var_it == inputs.end()) {
    std::abort();  // TODO also log an error
  }
  return var_it->second;
}

float Var::eval_grad(const Inputs& inputs,
                     Eigen::Map<Eigen::RowVectorXf>& grad_out) const noexcept {
  // derivatives of a variable w.r.t. all variables is a one-hot vector:
  // the only 1. is at the position of the variable itself
  for (std::size_t i = 0; i < inputs.size(); ++i) grad_out[i] = 0;
  // TODO find a way to move this out of the hot loop
  const std::size_t var_idx = find_var_idx(name, inputs);
  grad_out[var_idx] = 1.;

  return eval(inputs);
}

gpb::Graph Var::to_proto() const noexcept {
  gpb::Var var;
  var.set_name(name);

  gpb::Graph ret;
  *ret.mutable_var() = std::move(var);

  return ret;
}

std::unique_ptr<Var> Var::from_proto(const gpb::Var& vproto) noexcept {
  return std::make_unique<Var>(vproto.name());
}

gpb::Graph Graph::to_proto() const noexcept { return op->to_proto(); }

Graph Graph::from_proto(const gpb::Graph& gproto) noexcept {
  return Graph(op_from_proto(gproto));
}

absl::Status compute_graph_ad::to_file(const Graph& graph, fs::path path) {
  absl::Status ret_status = absl::OkStatus();

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  const gpb::Graph gproto = graph.to_proto();

  {
    std::ofstream out_file(path);
    if (!out_file.good()) {
      return absl::InvalidArgumentError(
          fmt::format("Could not open file {} for writing.", path.string()));
    }
    const bool ok = gproto.SerializeToOstream(&out_file);
    if (!ok)
      ret_status.Update(absl::AbortedError(
          fmt::format("Something went wrong while serializing Graph to file {}",
                      path.string())));
  }

  return ret_status;
}

absl::StatusOr<Graph> compute_graph_ad::from_file(fs::path path) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  gpb::Graph gproto;

  {
    std::ifstream in_file(path);
    if (!in_file.good()) {
      return absl::InvalidArgumentError(
          fmt::format("Could not open file {} for reading.", path.string()));
    }

    const bool ok = gproto.ParseFromIstream(&in_file);
    if (!ok) {
      return absl::AbortedError(
          fmt::format("Something went wrong while serializing Graph to file {}",
                      path.string()));
    }
  }

  absl::StatusOr<Graph> ret = Graph::from_proto(gproto);
  return ret;
}
