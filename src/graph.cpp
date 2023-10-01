/*
cpp-graph-autodiff  Copyright (C) 2023 Enrico Guiraud
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions: see LICENSE.
*/
#include "graph.h"

#include <cassert>
#include <fstream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fmt/core.h"
#include "src/graph.pb.h"

using namespace compute_graph_ad;
namespace gpb = graph_proto;

namespace {
gpb::Graph* make_proto_graph(gpb::Graph::OpCase optype, void* opptr) {
  auto* gproto = new gpb::Graph();  // caller will take ownership

  switch (optype) {
    case gpb::Graph::OpCase::kSum:
      gproto->set_allocated_sum(static_cast<gpb::Sum*>(opptr));
      break;
    case gpb::Graph::OpCase::kMul:
      gproto->set_allocated_mul(static_cast<gpb::Mul*>(opptr));
      break;
    case gpb::Graph::OpCase::kVar:
      gproto->set_allocated_var(static_cast<gpb::Var*>(opptr));
      break;
    case gpb::Graph::OpCase::kConst:
      gproto->set_allocated_const_(static_cast<gpb::Const*>(opptr));
      break;
    case gpb::Graph::OpCase::OP_NOT_SET:
      std::abort();  // TODO and log: this should never happen
      break;
  }

  return gproto;
}

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
}  // end of anonymous namespace

float Sum::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) + op2->eval(inputs);
}

std::pair<gpb::Graph::OpCase, void*> Sum::to_proto() const noexcept {
  auto* sum = new gpb::Sum();  // caller will take ownership

  // op1
  auto [op1type, op1ptr] = op1->to_proto();
  gpb::Graph* g1 = make_proto_graph(op1type, op1ptr);
  sum->set_allocated_op1(g1);

  // op2
  auto [op2type, op2ptr] = op2->to_proto();
  gpb::Graph* g2 = make_proto_graph(op2type, op2ptr);
  sum->set_allocated_op2(g2);

  return {gpb::Graph::OpCase::kSum, sum};
}

std::unique_ptr<Sum> Sum::from_proto(const gpb::Sum& sproto) noexcept {
  return std::make_unique<Sum>(op_from_proto(sproto.op1()),
                               op_from_proto(sproto.op2()));
}

float Mul::eval(const Inputs& inputs) const noexcept {
  assert(op1 && op2);
  return op1->eval(inputs) * op2->eval(inputs);
}

std::pair<gpb::Graph::OpCase, void*> Mul::to_proto() const noexcept {
  auto* mul = new gpb::Mul();  // caller will take ownership

  // op1
  auto [op1type, op1ptr] = op1->to_proto();
  gpb::Graph* g1 = make_proto_graph(op1type, op1ptr);
  mul->set_allocated_op1(g1);

  // op2
  auto [op2type, op2ptr] = op2->to_proto();
  gpb::Graph* g2 = make_proto_graph(op2type, op2ptr);
  mul->set_allocated_op2(g2);

  return {gpb::Graph::OpCase::kMul, mul};
}

std::unique_ptr<Mul> Mul::from_proto(const gpb::Mul& mproto) noexcept {
  return std::make_unique<Mul>(op_from_proto(mproto.op1()),
                               op_from_proto(mproto.op2()));
}

float Graph::eval(const Inputs& inputs) const noexcept {
  assert(op);
  return op->eval(inputs);
}

std::pair<gpb::Graph::OpCase, void*> Const::to_proto() const noexcept {
  auto* c = new gpb::Const();  // caller will take ownership
  c->set_value(value);
  return {gpb::Graph::OpCase::kConst, c};
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

std::pair<gpb::Graph::OpCase, void*> Var::to_proto() const noexcept {
  auto* var = new gpb::Var();  // caller will take ownership
  var->set_name(name);
  return {gpb::Graph::OpCase::kVar, var};
}

std::unique_ptr<Var> Var::from_proto(const gpb::Var& vproto) noexcept {
  return std::make_unique<Var>(vproto.name());
}

std::unique_ptr<const gpb::Graph> Graph::to_proto() const noexcept {
  auto [optype, opptr] = op->to_proto();
  std::unique_ptr<const gpb::Graph> gproto(make_proto_graph(optype, opptr));
  return gproto;
}

Graph Graph::from_proto(const gpb::Graph& gproto) noexcept {
  return Graph(op_from_proto(gproto));
}

absl::Status compute_graph_ad::to_file(const Graph& graph, fs::path path) {
  absl::Status ret_status = absl::OkStatus();

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  const std::unique_ptr<const gpb::Graph> gproto = graph.to_proto();

  {
    std::ofstream out_file(path);
    if (!out_file.good()) {
      return absl::InvalidArgumentError(
          fmt::format("Could not open file {} for writing.", path.string()));
    }
    const bool ok = gproto->SerializeToOstream(&out_file);
    if (!ok)
      ret_status.Update(absl::AbortedError(
          fmt::format("Something went wrong while serializing Graph to file {}",
                      path.string())));
  }

  google::protobuf::ShutdownProtobufLibrary();

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
      google::protobuf::ShutdownProtobufLibrary();
      return absl::AbortedError(
          fmt::format("Something went wrong while serializing Graph to file {}",
                      path.string()));
    }
  }

  absl::StatusOr<Graph> ret = Graph::from_proto(gproto);
  return ret;
}
