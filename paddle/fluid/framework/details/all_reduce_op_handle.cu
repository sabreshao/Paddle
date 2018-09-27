//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(allreduce_check, false, "If set, check allreduce result.");
DEFINE_bool(allreduce_single_stream, true, "Batch size of input data");
DEFINE_bool(allreduce_use_cpu, false, "use cpu to perform allreduce");
DEFINE_int32(allreduce_thread, 512, "Batch size of input data");
DEFINE_int32(allreduce_grid, 64, "Batch size of input data");

namespace paddle {
namespace framework {
namespace details {

template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] += B[idx];
  }
}

template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B, T* C) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] ;
  }
}

template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B,T* C, T* D) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] + D[idx];
  }
}
template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B,T* C, T* D, T* E) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] + D[idx] + E[idx];
  }
}
template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B,T* C, T* D,
 T* E, T* F) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] + D[idx] + E[idx] + F[idx];
  }
}
template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B,T* C, T* D,
 T* E, T* F,T* G) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] + D[idx] + E[idx] + F[idx] + G[idx];
  }
}
template <typename T>
__global__ void allreduce_sum(size_t lens, T* A, T* B,T* C, T* D,
 T* E, T* F,T* G, T* H) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < lens ; idx += blockDim.x * gridDim.x)
  {
    A[idx] = A[idx] + B[idx] + C[idx] + D[idx] + E[idx] + F[idx] + G[idx] + H[idx];
  }
}
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLContextMap *ctxs)
    : OpHandleBase(node),
      local_scopes_(local_scopes),
      places_(places),
      nccl_ctxs_(ctxs) {
  if (nccl_ctxs_) {
    for (auto &p : places_) {
      this->dev_ctxes_[p] = nccl_ctxs_->DevCtx(p);
    }
  }
}
#else
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}
#endif

void AllReduceOpHandle::RunImpl() {
  platform::RecordEvent r("all_reduce", nullptr);
  if (NoDummyInputSize() == 1) {
    return;  // No need to all reduce when GPU count = 1;
  } else {
    // Wait input done
    WaitInputVarGenerated();
    auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
    auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
    PADDLE_ENFORCE_EQ(
        in_var_handles.size(), places_.size(),
        "The NoDummyInputSize should be equal to the number of places.");
    PADDLE_ENFORCE_EQ(
        in_var_handles.size(), out_var_handles.size(),
        "The NoDummyInputSize and NoDummyOutputSize should be equal.");

    std::vector<const LoDTensor *> lod_tensors;
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto *s = local_scopes_[i];
      auto &local_scope = *s->FindVar(kLocalExecScopeName)->Get<Scope *>();
      auto &lod_tensor =
          local_scope.FindVar(in_var_handles[i]->name_)->Get<LoDTensor>();
      lod_tensors.emplace_back(&lod_tensor);
      PADDLE_ENFORCE_EQ(in_var_handles[i]->name_, out_var_handles[i]->name_,
                        "The name of input and output should be equal.");
    }

    VLOG(1) << "device num " << local_scopes_.size() << " size " << lod_tensors[0]->numel();
    if (platform::is_gpu_place(lod_tensors[0]->place())) {
      const bool use_cpu =  FLAGS_allreduce_use_cpu || (local_scopes_.size() > 8);
      const bool check = FLAGS_allreduce_check && !use_cpu;
      const int threads = FLAGS_allreduce_thread;
      const int grid = FLAGS_allreduce_grid;
#if defined(PADDLE_WITH_HIP)
      size_t numel = lod_tensors[0]->numel();
      std::vector<float*> buffers;
      std::vector<hipStream_t> streams;
      framework::Tensor output_cpu[local_scopes_.size()];
      hipEvent_t events[local_scopes_.size()];
      framework::Tensor final_output_cpu;
      for (size_t i = 0; i < local_scopes_.size(); ++i) {
        auto &p = places_[i];
        auto &lod_tensor = *lod_tensors[i];
        float* buffer = const_cast<float *>(lod_tensor.data<float>());
        buffers.emplace_back(buffer);

        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);
        //auto stream = nccl_ctx.stream();
        auto stream = static_cast<platform::CUDADeviceContext *>(dev_ctxes_[p])->stream();
        if (FLAGS_allreduce_single_stream)
        {
          streams.emplace_back(stream);
        }
        else
        {
          streams.emplace_back(nccl_ctx.stream());
          hipStreamSynchronize(stream);
        }

        if (use_cpu || check)
        {
          output_cpu[i].mutable_data<float>(lod_tensor.dims(), platform::CPUPlace());
          framework::TensorCopySync(lod_tensor, platform::CPUPlace(), &(output_cpu[i]));
        }
      }
      if (use_cpu)
      {
        //merge all original data to final_output_cpu at CPU side
        final_output_cpu.mutable_data<float>(lod_tensors[0]->dims(), platform::CPUPlace());
        for (size_t i = 0; i < local_scopes_.size(); ++i) {
          for (int j = 0; j < output_cpu[i].numel() ; j++)
            if(i == 0)
              (final_output_cpu.data<float>())[j] =  (output_cpu[i].data<float>())[j];
            else
              (final_output_cpu.data<float>())[j] +=  (output_cpu[i].data<float>())[j];
        }

        //copy merged data back to different GPUs
        for (size_t i = 0; i < local_scopes_.size(); ++i) {
          auto &lod_tensor = const_cast<framework::LoDTensor&>(*lod_tensors[i]);
          auto &p = places_[i];

          framework::TensorCopySync(final_output_cpu, p, &lod_tensor);
        }
      }
      else {

        static int ring_index = 0 ;
        int idx[local_scopes_.size()];

        for (size_t i = 0; i < local_scopes_.size(); ++i){
          idx[i] = (ring_index+i) % local_scopes_.size();
          hipEventCreate(&events[i]);
        }

        /*sync all stream*/
        for (size_t i = 1; i < local_scopes_.size(); ++i){
          hipEventRecord(events[idx[i]], streams[idx[i]]);
          hipStreamWaitEvent(streams[idx[0]], events[idx[i]], 0);
        }

        if( local_scopes_.size() == 2 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]]);
        else if( local_scopes_.size() == 3 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]]);
        else if( local_scopes_.size() == 4 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]], buffers[idx[3]]);
        else if( local_scopes_.size() == 5 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]], buffers[idx[3]],
                              buffers[idx[4]]);
        else if( local_scopes_.size() == 6 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]], buffers[idx[3]],
                              buffers[idx[4]], buffers[idx[5]]);
        else if( local_scopes_.size() == 7 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]], buffers[idx[3]],
                              buffers[idx[4]], buffers[idx[5]], buffers[idx[6]]);
        else if( local_scopes_.size() == 8 )
          hipLaunchKernelGGL((allreduce_sum<float>), dim3(grid), dim3(threads), 0, streams[idx[0]], numel,
                              buffers[idx[0]], buffers[idx[1]], buffers[idx[2]], buffers[idx[3]],
                              buffers[idx[4]], buffers[idx[5]], buffers[idx[6]], buffers[idx[7]]);

        /*broadcast results to all gpus */

        for (size_t dst_i = 1, src_i = 0, ready_count = 1; dst_i < local_scopes_.size(); ++dst_i){
          hipMemcpyAsync(buffers[idx[dst_i]], buffers[idx[src_i]], numel * sizeof(float), hipMemcpyDeviceToDevice, streams[idx[src_i]]);
          hipEventRecord(events[idx[dst_i]], streams[idx[src_i]]);
          hipStreamWaitEvent(streams[idx[dst_i]], events[idx[dst_i]], 0);
          if((--ready_count) == 0){
            ready_count = dst_i+1;
            src_i = 0;
          }
          else src_i++;
        }

        ring_index = (ring_index+1) % local_scopes_.size();

        if (!FLAGS_allreduce_single_stream)
          for (size_t i = 0; i < local_scopes_.size(); ++i)
            hipStreamSynchronize(streams[i]);

        for (size_t i = 0; i < local_scopes_.size(); ++i)
          hipEventDestroy(events[i]);
      }
      if (check)
      {
          final_output_cpu.mutable_data<float>(lod_tensors[0]->dims(), platform::CPUPlace());
          for (int k = 0; k < local_scopes_.size(); ++k)
          {
              VLOG(1) << "checking " << k;
              VLOG(1) << " org " << (output_cpu[k].data<float>())[0];
              framework::TensorCopySync(*lod_tensors[k], platform::CPUPlace(), &(final_output_cpu));
              for (int j = 0; j < numel ; j++) {
                  float temp = 0.0f;
    for (size_t i = 0; i < local_scopes_.size(); ++i)
                      temp +=  (output_cpu[i].data<float>())[j];
                  if ((j == 0) || ( temp - (final_output_cpu.data<float>())[j] > 1e-3 || temp - (final_output_cpu.data<float>())[j] < -1e-3))
                      VLOG(1) << "data " << temp << " " << (final_output_cpu.data<float>())[j];
              }
          }
      }
#else
      PADDLE_THROW("Not compiled with CUDA");
#endif
    } else {  // Special handle CPU only Operator's gradient. Like CRF
      auto &trg = *this->local_scopes_[0]
                       ->FindVar(kLocalExecScopeName)
                       ->Get<Scope *>()
                       ->FindVar(out_var_handles[0]->name_)
                       ->GetMutable<framework::LoDTensor>();

      // Reduce All Tensor to trg in CPU
      ReduceLoDTensor func(lod_tensors, &trg);
      VisitDataType(ToDataType(lod_tensors[0]->type()), func);

      for (size_t i = 1; i < local_scopes_.size(); ++i) {
        auto &scope =
            *local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope *>();
        auto &p = places_[i];
        auto *var = scope.FindVar(out_var_handles[i]->name_);
        auto *dev_ctx = dev_ctxes_[p];

        RunAndRecordEvent(p, [&trg, var, dev_ctx, p] {
          auto &tensor_gpu = *var->GetMutable<framework::LoDTensor>();
          auto &tensor_cpu = trg;
          TensorCopy(tensor_cpu, p, *dev_ctx, &tensor_gpu);
        });
      }
    }
  }
}

std::string AllReduceOpHandle::Name() const { return "all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
