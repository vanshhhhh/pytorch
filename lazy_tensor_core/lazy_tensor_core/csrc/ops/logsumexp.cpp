#include "lazy_tensor_core/csrc/ops/logsumexp.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Logsumexp::Logsumexp(const torch::lazy::Value& input,
                     std::vector<int64_t> dimensions,
                     bool keep_reduced_dimensions)
    : TsNode(torch::lazy::OpKind(at::aten::logsumexp), {input},
             /*num_outputs=*/1,
             torch::lazy::MHash(dimensions, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Logsumexp::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Logsumexp>(operands.at(0), dimensions_,
                             keep_reduced_dimensions_);
}

std::string Logsumexp::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dimensions=(" << c10::Join(", ", dimensions_)
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors