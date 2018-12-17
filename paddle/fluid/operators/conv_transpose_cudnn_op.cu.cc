/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/fluid/platform/assert.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#else
#include "paddle/fluid/platform/miopen_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static constexpr size_t kConvCUDNNWorkspaceLimitBytes = 1024 * 1024 * 1024;

template <typename T>
class CUDNNConvTransposeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout;

    if (strides.size() == 2U) {
      layout = DataLayout::kNCHW;
    } else {
      layout = DataLayout::kNCDHW;
    }

#ifdef PADDLE_WITH_CUDA
    // (N, M, H, W) or (N, M, D, H, W)
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    // (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()), groups);
    // (M, C, K_h, K_w) or (M, C, K_d, K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);
#else  // PADDLE_WITH_CUDA
    // (N, M, H, W) or (N, M, D, H, W)
    miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), 1);
    // (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()), 1);
    // (M, C, K_h, K_w) or (M, C, K_d, K_h, K_w)
    miopenTensorDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), 1);
    miopenConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);
    // must set group count
    PADDLE_ENFORCE(platform::dynload::miopenSetConvolutionGroupCount(	
            cudnn_conv_desc, groups));
    groups = 1;
#endif

    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }
#ifdef PADDLE_WITH_CUDA
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t algo;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    // Get the algorithm
    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
        handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
        // dxDesc: Handle to the previously initialized output tensor
        // descriptor.
        cudnn_output_desc, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_size_limit, &algo));

    // get workspace size able to allocate
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, algo, &workspace_size_in_bytes));

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_offset = output->numel() / output->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    for (int g = 0; g < groups; g++) {
      auto cudnn_func = [&](void* cudnn_workspace) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
            handle, &alpha, cudnn_filter_desc, filter_data + filter_offset * g,
            cudnn_input_desc, input_data + input_offset * g, cudnn_conv_desc,
            algo, cudnn_workspace, workspace_size_in_bytes, &beta,
            cudnn_output_desc, output_data + output_offset * g));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
    }
#else  // PADDLE_WITH_CUDA
    (void)workspace_size_limit;
    miopenConvBwdDataAlgorithm_t algo;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.miopen_handle();

    miopenConvAlgoPerf_t perfRes;
    int algoCount = 0;

    // get workspace size able to allocate
    PADDLE_ENFORCE(
        platform::dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, &workspace_size_in_bytes));

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_offset = output->numel() / output->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    auto workspace_handle = dev_ctx.miopen_workspace_handle();
    for (int g = 0; g < groups; g++) {
      auto miopen_func_algo = [&](void * miopen_workspace){
        PADDLE_ENFORCE(platform::dynload::miopenFindConvolutionBackwardDataAlgorithm(
            handle, cudnn_input_desc, input_data,cudnn_filter_desc, filter_data, cudnn_conv_desc,
            // dxDesc: Handle to the previously initialized output tensor
            // descriptor.
            cudnn_output_desc, output_data,1,&algoCount, &perfRes, miopen_workspace, workspace_size_in_bytes,false));
      };
      workspace_handle.RunFunc(miopen_func_algo, workspace_size_in_bytes);
      algo=perfRes.bwd_data_algo;
      auto miopen_func = [&](void* miopen_workspace) {
        PADDLE_ENFORCE(platform::dynload::miopenConvolutionBackwardData(
            handle, &alpha, cudnn_input_desc, input_data + input_offset * g,
            cudnn_filter_desc, filter_data + filter_offset * g,
            cudnn_conv_desc, algo, &beta, cudnn_output_desc, output_data + output_offset * g, miopen_workspace,
            workspace_size_in_bytes));
      };
      workspace_handle.RunFunc(miopen_func, workspace_size_in_bytes);
    }
#endif
  }
};

template <typename T>
class CUDNNConvTransposeGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;

#ifdef PADDLE_WITH_CUDA
    // Input: (N, M, H, W) or (N, M, D, H, W)
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    // Output: (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output_grad->dims()), groups);
    // Filter (M, C, K_h, K_w) or (M, C, K_d K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);
#else  // PADDLE_WITH_CUDA
    // Input: (N, M, H, W) or (N, M, D, H, W)
    miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), 1);
    // Output: (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output_grad->dims()), 1);
    // Filter (M, C, K_h, K_w) or (M, C, K_d K_h, K_w)
    miopenTensorDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), 1);

    miopenConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // must set group count
    PADDLE_ENFORCE(platform::dynload::miopenSetConvolutionGroupCount(	
            cudnn_conv_desc, groups));
    groups = 1;
#endif

#ifdef PADDLE_WITH_CUDA
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionFwdAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t bwd_filter_ws_size, fwd_ws_size;
    size_t workspace_size_in_bytes = 0;
    size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    if (input_grad) {
      // choose backward algorithm for data
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &data_algo));
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, data_algo, &fwd_ws_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, fwd_ws_size);
    }

    if (filter_grad) {
      // choose backward algorithm for filter
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc,
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &filter_algo));

      // get workspace for backwards filter algorithm
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc, filter_algo, &bwd_filter_ws_size));
      workspace_size_in_bytes =
          std::max(workspace_size_in_bytes, bwd_filter_ws_size);
    }

    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_grad_offset =
        output_grad->numel() / output_grad->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
              handle, &alpha, cudnn_output_desc,
              output_grad_data + output_grad_offset * g, cudnn_filter_desc,
              filter_data + filter_offset * g, cudnn_conv_desc, data_algo,
              cudnn_workspace, workspace_size_in_bytes, &beta, cudnn_input_desc,
              input_grad_data + input_offset * g));
        };
        workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
              handle, &alpha, cudnn_output_desc,
              output_grad_data + output_grad_offset * g, cudnn_input_desc,
              input_data + input_offset * g, cudnn_conv_desc, filter_algo,
              cudnn_workspace, workspace_size_in_bytes, &beta,
              cudnn_filter_desc, filter_grad_data + filter_offset * g));
        };
        workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      }
    }
#else // PADDLE_WITH_CUDA
    miopenConvFwdAlgorithm_t data_algo = miopenConvolutionFwdAlgoGEMM;
    miopenConvBwdWeightsAlgorithm_t filter_algo = miopenConvolutionBwdWeightsAlgoGEMM;

    size_t bwd_filter_ws_size, fwd_ws_size;
    size_t workspace_size_in_bytes = 0;
    size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }
    (void)workspace_size_limit;


    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.miopen_handle();
    miopenConvAlgoPerf_t perfRes;
    auto workspace_handle = dev_ctx.miopen_workspace_handle();
    int algoCount = 0;
    if (input_grad) {
      PADDLE_ENFORCE(platform::dynload::miopenConvolutionForwardGetWorkSpaceSize(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, &fwd_ws_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, fwd_ws_size);
      auto miopen_func_algo = [&](void * miopen_workspace){
          PADDLE_ENFORCE(platform::dynload::miopenFindConvolutionForwardAlgorithm(
            handle, cudnn_output_desc, (const void*)output_grad_data, cudnn_filter_desc, 
            (const void*)filter_data,cudnn_conv_desc, cudnn_input_desc, (void*)input_data,
            1, &algoCount, &perfRes, miopen_workspace, workspace_size_in_bytes, false));
      };
      workspace_handle.RunFunc(miopen_func_algo, workspace_size_in_bytes);
      data_algo=perfRes.fwd_algo;
    }

    if (filter_grad) {
      PADDLE_ENFORCE(
          platform::dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
              handle, cudnn_input_desc, cudnn_output_desc, cudnn_conv_desc,
              cudnn_filter_desc, &bwd_filter_ws_size));
      workspace_size_in_bytes =
          std::max(workspace_size_in_bytes, bwd_filter_ws_size);
      auto miopen_func_algo = [&](void* miopen_workspace) {
        PADDLE_ENFORCE(
        platform::dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
            handle,
            cudnn_input_desc, (const void*)input_data,
            cudnn_output_desc, (const void*)output_grad_data,
            cudnn_conv_desc,
            cudnn_filter_desc, (void*)filter_data,
            1, &algoCount, &perfRes,
            miopen_workspace, workspace_size_in_bytes, false));
      };
      workspace_handle.RunFunc(miopen_func_algo, workspace_size_in_bytes);
      filter_algo=perfRes.bwd_weights_algo;
    }

    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_grad_offset =
        output_grad->numel() / output_grad->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int g = 0; g < groups; g++) {
        auto miopen_func = [&](void* miopen_workspace) {
          PADDLE_ENFORCE(platform::dynload::miopenConvolutionForward(
              handle, &alpha, cudnn_output_desc, output_grad_data + output_grad_offset * g,
              cudnn_filter_desc, filter_data + filter_offset * g, cudnn_conv_desc, data_algo,
              &beta, cudnn_input_desc, input_grad_data + input_offset * g, miopen_workspace, 
              workspace_size_in_bytes));
        };
        workspace_handle.RunFunc(miopen_func, workspace_size_in_bytes);
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
        auto miopen_func = [&](void* miopen_workspace) {
          PADDLE_ENFORCE(platform::dynload::miopenConvolutionBackwardWeights(
              handle, &alpha, cudnn_input_desc, input_data + input_offset * g,
              cudnn_output_desc, output_grad_data + output_grad_offset * g,
              cudnn_conv_desc, filter_algo, &beta, cudnn_filter_desc,
              filter_grad_data + filter_offset * g,
              miopen_workspace, workspace_size_in_bytes));
        };
        workspace_handle.RunFunc(miopen_func, workspace_size_in_bytes);
      }
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>);
