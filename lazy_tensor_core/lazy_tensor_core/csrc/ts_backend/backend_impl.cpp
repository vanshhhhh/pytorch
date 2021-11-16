#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"

#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace compiler {

class TSBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const torch::lazy::BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<TSData>(data);
    return ts_data->data();
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device) const override {
    at::TensorOptions options = tensor.options().device(GetDefaultDeviceType());
    return std::make_shared<TSData>(tensor.to(options), shape, device);
  }

  std::string GetComputationBackendText(
      const torch::lazy::ComputationPtr computation) const override {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(computation.get());
    return ts_computation->graph()->toString();
  }

  //////////////computation client interfaces///////////////////////

 public:
  class TSData : public torch::lazy::BackendData {
   public:
    TSData(const at::Tensor& data, const torch::lazy::Shape& shape,
           const torch::lazy::BackendDevice& device)
        : torch::lazy::BackendData(device, shape), data_(data) {}

    TSData(const torch::lazy::Shape& shape,
           const torch::lazy::BackendDevice& device)
        : torch::lazy::BackendData(device, shape) {}

    Handle GetHandle() override { return reinterpret_cast<int64_t>(this); }

    void Assign(const torch::lazy::BackendData& data) override {
      data_ = static_cast<const TSData&>(data).data_;
    }

    bool HasValue() const override { return data_.defined(); }

    at::Tensor data() { return data_; }

   private:
    at::Tensor data_;
  };

  torch::lazy::BackendDataPtr CreateDataPlaceholder(
      const torch::lazy::BackendDevice& device,
      const torch::lazy::Shape& shape) const override;

  std::vector<torch::lazy::ComputationPtr> Compile(
      std::vector<torch::lazy::ComputationPtr> instances) const override;

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      torch::lazy::Computation& computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const override;

  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType() const override;

  void SetDefaultDeviceType(std::string) override {}

  at::DeviceType EagerFallbackDeviceType() const override;

  std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override;

  torch::lazy::BackendDevice GetBackendDevice(c10::Device device) const override;

  void SetRngSeed(size_t seed) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  // std::map<std::string, Metric> GetMetrics() const override { return {}; }

  // MemoryInfo GetMemoryInfo(const std::string& device) override {
  //   LOG(FATAL) << "Not implemented yet.";
  // }

  void PrepareToExit() const override;
};

torch::lazy::BackendDataPtr TSBackendImpl::CreateDataPlaceholder(
    const torch::lazy::BackendDevice& device,
    const torch::lazy::Shape& shape) const {
  return std::make_shared<TSBackendImpl::TSData>(shape, device);
}

std::vector<torch::lazy::ComputationPtr> TSBackendImpl::Compile(
    std::vector<torch::lazy::ComputationPtr> instances) const {
  for (const auto& instance : instances) {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(instance.get());
  }
  return instances;
}

std::vector<torch::lazy::BackendDataPtr> TSBackendImpl::ExecuteComputation(
    torch::lazy::Computation& computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) const {
  torch::jit::GraphExecutor& graph_executor =
      static_cast<torch::lazy::TSComputation&>(computation).graph_executor();
  std::vector<torch::jit::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data =
        std::static_pointer_cast<TSBackendImpl::TSData>(argument);
    CHECK(GetDefaultDeviceType()->type() != at::kCUDA ||
          ts_data->data().device().type() == at::kCUDA);
    stack.emplace_back(ts_data->data());
  }
  graph_executor.run(stack);
  std::vector<torch::lazy::BackendDataPtr> results;
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    torch::lazy::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(
        std::make_shared<TSBackendImpl::TSData>(result, shape, device));
  }
  return results;
}

torch::lazy::BackendDevice GetDefaultBackendDevice() {
  static c10::DeviceType device_type =
      lazy_tensors::sys_util::GetEnvBool("LTC_TS_CUDA", false) ? at::kCUDA
                                                               : at::kCPU;
  static torch::lazy::BackendDevice backend_device(BackendDeviceType(device_type), 0);
  // The first CUDA usage could happen via lazy tensors. Initialize CUDA here to
  // account for that, at::scalar_tensor constructor triggers everything we
  // need.
  static c10::optional<at::Tensor> init_cuda =
      device_type == at::kCUDA ? c10::optional<at::Tensor>(at::scalar_tensor(
                                     0, at::TensorOptions().device(at::kCUDA)))
                               : c10::nullopt;
  return backend_device;
}

std::shared_ptr<torch::lazy::BackendDeviceType> TSBackendImpl::GetDefaultDeviceType() const {
  return GetDefaultBackendDevice().type();
}

at::DeviceType TSBackendImpl::EagerFallbackDeviceType() const {
  return GetDefaultBackendDevice().type();
}

std::vector<torch::lazy::BackendDevice> TSBackendImpl::GetBackendDevices() const {
  return {GetDefaultBackendDevice()};
}

BackendDevice TSBackendImpl::GetBackendDevice(c10::Device device) const {
  return BackendDevice(device.type(), device.index());
}

void TSBackendImpl::PrepareToExit() const {}

torch::lazy::BackendImplInterface* GetTSBackendImpl() {
  static compiler::TSBackendImpl* ts_backend_impl =
      new compiler::TSBackendImpl();
  return ts_backend_impl;
}

void InitTorchScriptBackend() {
  static std::unique_ptr<torch::lazy::BackendRegistrar> s_registrar;
  s_registrar.reset(
      new torch::lazy::BackendRegistrar(compiler::GetTSBackendImpl()));
}
};  // namespace compiler
}  // namespace torch_lazy_tensors
