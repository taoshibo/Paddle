#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "unsupported/Eigen/CXX11/Tensor"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/external_kernels/register.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fc_op.cc"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/activation_op.h"

#include "cinn/hlir/framework/buffer.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/tensor/dense_host_tensor.h"
#include "cinnrt/common/global.h"

using Scope = paddle::framework::Scope;
using VariableNameMap = paddle::framework::VariableNameMap;
using AttributeMap = paddle::framework::AttributeMap;
using Tensor = paddle::framework::Tensor;
using Place = paddle::platform::Place;
using CPUPlace = paddle::platform::CPUPlace;
using FCOp = paddle::operators::FCOp;
using RuntimeContext = paddle::framework::RuntimeContext;
using DenseHostTensor = cinnrt::tensor::DenseHostTensor;
using DeviceContext = paddle::platform::DeviceContext;
using DeviceContextPool = paddle::platform::DeviceContextPool;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;
using ExecutionContext = paddle::framework::ExecutionContext;
using FCOpKernel = paddle::operators::FCOpKernel<CPUDeviceContext, float>;
using Attribute_INT = cinnrt::host_context::Attribute<int>;
using paddle::platform::Transform;
using paddle::operators::math::ReluFunctor;
using paddle::operators::math::SigmoidFunctor;

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace math = paddle::operators::math;
namespace jit = paddle::operators::jit;

//template<typename T> 
//void fc(const DenseHostTensor &input, DenseHostTensor *w, DenseHostTensor *bias, DenseHostTensor *output, cinnrt::host_context::Attribute<int> in_num_col_dims) {
//    std::cout << "input: " << input << std::endl;
//    std::cout << "w: " << *w << std::endl;
//    std::cout << "bias: " << *bias << std::endl;
//    std::cout << "output: " << *output << std::endl;
//    std::cout << "in_num_col_dims: " << in_num_col_dims.get() << std::endl;
//}

static Tensor dht_to_tensor(const DenseHostTensor &src, Place place) {
    std::cout << "dht_to_tensor begin" << std::endl;
    auto* data = reinterpret_cast<float*>(src.buffer()->data()->memory);
    int num_elements = src.shape().GetNumElements();
    for (int i = 0; i < num_elements - 1; i++) std::cout << data[i] << ", ";
    if (num_elements > 0) std::cout << data[num_elements - 1] << std::endl;;
    std::cout << src << std::endl;

    std::vector<int64_t> dims;
    auto shape = src.shape();
    for (int i = 0; i < shape.GetRank(); ++i) dims.push_back(shape.GetDim(i));
    Tensor dst;
    float* p = dst.mutable_data<float>(framework::make_ddim(dims), place);
    for (int i = 0; i < shape.GetNumElements(); ++i) *(p + i) = data[i];
    return dst;
}

//template<typename T> 
//void fc(const DenseHostTensor &input, DenseHostTensor *w, DenseHostTensor *bias, DenseHostTensor *out, cinnrt::host_context::Attribute<int> in_num_col_dims, cinnrt::host_context::Attribute<int> test_attr) {
//    std::cout << "input: " << input << std::endl;
//    std::cout << "w: " << *w << std::endl;
//    std::cout << "bias: " << *bias << std::endl;
//    std::cout << "out: " << *out << std::endl;
//    std::cout << "in_num_col_dims: " << in_num_col_dims.get() << std::endl;
//    std::cout << "test_attr: " << test_attr.get() << std::endl;
//
//    auto place = platform::CPUPlace();
//    DeviceContextPool::Init({place});
//    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
//    std::cout << dev_ctx << std::endl;
//
//    Tensor input_tensor = dht_to_tensor(input, place);
//    Tensor w_tensor = dht_to_tensor(*w, place);
//    Tensor bias_tensor = dht_to_tensor(*bias, place);
//    Tensor out_tensor = dht_to_tensor(*out, place);
//
//    std::cout << "----------------" << std::endl;
//    std::cout << "input_tensor: " << input_tensor << std::endl;
//    std::cout << "w_tensor: " << w_tensor << std::endl;;
//    std::cout << "bias_tensor: " << bias_tensor << std::endl;;
//    std::cout << "out_tensor: " << out_tensor << std::endl;;
//    std::cout << "================" << std::endl;
//
//    auto kernel = FCOpKernel();
//    std::cout << &kernel << std::endl;
//    auto *scope = new Scope();
//    VariableNameMap inputs = {};
//    VariableNameMap outputs = {};
//    AttributeMap attrs = {};
//
//    RuntimeContext rt_ctx(inputs, outputs, *scope);
//    FCOp fc_op("fc", inputs, outputs, attrs);
//    ExecutionContext exe_ctx(fc_op, *scope, *dev_ctx, rt_ctx);
//    std::cout << &exe_ctx << std::endl;
//    kernel.Compute(exe_ctx);
//}

template<typename T> 
void fc1(const DenseHostTensor &input, DenseHostTensor *w, DenseHostTensor *bias, DenseHostTensor *out, cinnrt::host_context::Attribute<int> in_num_col_dims, cinnrt::host_context::Attribute<int> test_attr) {
    std::cout << "input: " << input << std::endl;
    std::cout << "w: " << *w << std::endl;
    std::cout << "bias: " << *bias << std::endl;
    std::cout << "out: " << *out << std::endl;
    std::cout << "in_num_col_dims: " << in_num_col_dims.get() << std::endl;
    std::cout << "test_attr: " << test_attr.get() << std::endl;

    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    std::cout << dev_ctx << std::endl;
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    Tensor input_tensor = dht_to_tensor(input, place);
    Tensor w_tensor = dht_to_tensor(*w, place);
    Tensor bias_tensor = dht_to_tensor(*bias, place);
    Tensor out_tensor = dht_to_tensor(*out, place);

    std::cout << "----------------" << std::endl;
    std::cout << "input_tensor: " << input_tensor << std::endl;
    std::cout << "w_tensor: " << w_tensor << std::endl;;
    std::cout << "bias_tensor: " << bias_tensor << std::endl;;
    std::cout << "out_tensor: " << out_tensor << std::endl;;
    std::cout << "================" << std::endl;

    auto blas = math::GetBlas<CPUDeviceContext, float>(cpu_dev_ctx);
    int M = out_tensor.dims()[0];
    int N = out_tensor.dims()[1];
    int K = w_tensor.dims()[0];
    const float *X = input_tensor.data<float>();
    const float *W = w_tensor.data<float>();
    const float *B = bias_tensor.data<float>();
    float *Y = out_tensor.mutable_data<float>(place);
    blas.MatMul(M, N, K, X, W, Y);
    bool relu = true;
    auto compute = relu ?
            jit::KernelFuncs<jit::VAddReluTuple<float>, CPUPlace>::Cache().At(N)
            : jit::KernelFuncs<jit::VAddTuple<T>, CPUPlace>::Cache().At(N);
    for (int i = 0; i < M; i++) {
      float* dst = Y + i * N;
      float* src = dst;
      compute(B, src, dst, N);
    }
    std::cout << "out_tensor: " << out_tensor << std::endl;;
}

template<typename T> 
void fc2(const DenseHostTensor &input, DenseHostTensor *w, DenseHostTensor *bias, DenseHostTensor *out, cinnrt::host_context::Attribute<int> in_num_col_dims, cinnrt::host_context::Attribute<int> test_attr) {
    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    int M = out->shape().GetDim(0);
    int N = out->shape().GetDim(1);
    int K = w->shape().GetDim(0);
    const T *X = reinterpret_cast<T*>(input.buffer()->data()->memory);
    const T *W = reinterpret_cast<T*>(w->buffer()->data()->memory);
    const T *B = reinterpret_cast<T*>(bias->buffer()->data()->memory);
    T       *Y = reinterpret_cast<T*>(out->buffer()->data()->memory);

    auto blas = math::GetBlas<CPUDeviceContext, T>(cpu_dev_ctx);
    blas.MatMul(M, N, K, X, W, Y);
    bool relu = true;
    auto compute = relu ?
            jit::KernelFuncs<jit::VAddReluTuple<T>, CPUPlace>::Cache().At(N)
            : jit::KernelFuncs<jit::VAddTuple<T>, CPUPlace>::Cache().At(N);
    for (int i = 0; i < M; i++) {
      T* dst = Y + i * N;
      T* src = dst;
      compute(B, src, dst, N);
    }
}

template<typename T>
void matmul(const DenseHostTensor &x, const DenseHostTensor &y, DenseHostTensor *z) {
    //auto *benchmark_stats = cinnrt::Global::getBenchmarkStats();
    //benchmark_stats->StartRun();

    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    int M = z->shape().GetDim(0);
    int N = z->shape().GetDim(1);
    int K = y.shape().GetDim(0);
    const T *X_data = reinterpret_cast<T*>(x.buffer()->data()->memory);
    const T *Y_data = reinterpret_cast<T*>(y.buffer()->data()->memory);
    T       *Z_data = reinterpret_cast<T*>(z->buffer()->data()->memory);

    auto blas = math::GetBlas<CPUDeviceContext, T>(cpu_dev_ctx);
    blas.MatMul(M, N, K, X_data, Y_data, Z_data);

    //benchmark_stats->StopRun();
}

// vector to string.
template<typename T>
std::string vector2str(std::vector<T> &v) {
    std::ostringstream oss;
    oss << "[";
    size_t i = 0;
    for (; i < v.size() - 1; ++i) oss << v[i] << ", ";
    if (i == v.size() - 1) oss << v[i] << "]";
    return oss.str();
}

template<typename T>
void elementwise_add(const DenseHostTensor &x, const DenseHostTensor &y, DenseHostTensor *z, Attribute_INT axis/* = -1*/) {
    //auto *benchmark_stats = cinnrt::Global::getBenchmarkStats();
    //benchmark_stats->StartRun();

    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    int num_elements = x.shape().GetNumElements();
    std::vector<int64_t> dims_x;
    for (int i = 0; i < x.shape().GetRank(); ++i) dims_x.push_back(x.shape().GetDim(i));
    std::vector<int64_t> dims_y;
    for (int i = 0; i < y.shape().GetRank(); ++i) dims_y.push_back(y.shape().GetDim(i));

    const T *X_data = reinterpret_cast<T*>(x.buffer()->data()->memory);
    const T *Y_data = reinterpret_cast<T*>(y.buffer()->data()->memory);
    T       *Z_data = reinterpret_cast<T*>(z->buffer()->data()->memory);
    if (dims_x == dims_y) {
        auto blas = math::GetBlas<CPUDeviceContext, T>(cpu_dev_ctx);
        blas.VADD(num_elements, X_data, Y_data, Z_data);
    } else {
        std::string msg = "ERROR: dims mismatch: " + vector2str(dims_x) + " != " + vector2str(dims_y);
        throw std::logic_error(msg); 
    }

    //benchmark_stats->StopRun();
}


template<typename T>
void relu(const DenseHostTensor &x, DenseHostTensor *y) {
    //auto *benchmark_stats = cinnrt::Global::getBenchmarkStats();
    //benchmark_stats->StartRun();

    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    //std::vector<int64_t> dims_x;
    //for (int i = 0; i < x.shape().GetRank(); ++i) dims_x.push_back(x.shape().GetDim(i));
    T *X_data = reinterpret_cast<T*>(x.buffer()->data()->memory);
    T       *Y_data = reinterpret_cast<T*>(y->buffer()->data()->memory);

    // 纯 CPU 计算方式, 无优化
    //int num_elements = x.shape().GetNumElements();
    //Transform<CPUDeviceContext> trans;
    //trans(cpu_dev_ctx, X_data, X_data + num_elements,
    //      Y_data, ReluFunctor<T>());

    // 用 eigen 实现, 有优化
    using EigenDSizes = Eigen::DSizes<Eigen::DenseIndex, 1>;
    using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;

    EigenDSizes dims;
    dims[0] = x.shape().GetNumElements();
    EigenTensorMap X_eigen = EigenTensorMap(X_data, dims);
    EigenTensorMap Y_eigen = EigenTensorMap(Y_data, dims);

    auto *device = cpu_dev_ctx.eigen_device();
    paddle::operators::ReluFunctor<T> functor;
    functor(*device, X_eigen, Y_eigen);

    //benchmark_stats->StopRun();
}

template<typename T>
void sigmoid(const DenseHostTensor &x, DenseHostTensor *y) {
    //auto *benchmark_stats = cinnrt::Global::getBenchmarkStats();
    //benchmark_stats->StartRun();

    auto place = CPUPlace();
    DeviceContextPool::Init({place});
    auto* dev_ctx = DeviceContextPool::Instance().Get(place);
    auto &cpu_dev_ctx = *reinterpret_cast<const CPUDeviceContext*>(dev_ctx);

    //std::vector<int64_t> dims_x;
    //for (int i = 0; i < x.shape().GetRank(); ++i) dims_x.push_back(x.shape().GetDim(i));
    T *X_data = reinterpret_cast<T*>(x.buffer()->data()->memory);
    T       *Y_data = reinterpret_cast<T*>(y->buffer()->data()->memory);

    // 纯 CPU 计算方式, 无优化
    //int num_elements = x.shape().GetNumElements();
    //Transform<CPUDeviceContext> trans;
    //trans(cpu_dev_ctx, X_data, X_data + num_elements,
    //      Y_data, SigmoidFunctor<T>());

    // 用 eigen 实现, 有优化
    using EigenDSizes = Eigen::DSizes<Eigen::DenseIndex, 1>;
    using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;

    EigenDSizes dims;
    dims[0] = x.shape().GetNumElements();
    EigenTensorMap X_eigen = EigenTensorMap(X_data, dims);
    EigenTensorMap Y_eigen = EigenTensorMap(Y_data, dims);

    auto *device = cpu_dev_ctx.eigen_device();
    paddle::operators::SigmoidFunctor<T> functor;
    functor(*device, X_eigen, Y_eigen);

    //benchmark_stats->StopRun();
}

void RegisterPaddleKernels(cinnrt::host_context::KernelRegistry *registry) {
  // float32 
  registry->AddKernel("external.fc1", CINN_KERNEL(fc1<float>));
  registry->AddKernel("external.fc2", CINN_KERNEL(fc2<float>));
  registry->AddKernel("external.matmul", CINN_KERNEL(matmul<float>));
  registry->AddKernel("external.elementwise_add", CINN_KERNEL(elementwise_add<float>));
  registry->AddKernel("external.relu", CINN_KERNEL(relu<float>));
  registry->AddKernel("external.sigmoid", CINN_KERNEL(sigmoid<float>));
}
