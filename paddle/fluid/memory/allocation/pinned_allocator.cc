// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

namespace paddle {
namespace memory {
namespace allocation {
bool CPUPinnedAllocator::IsAllocThreadSafe() const { return true; }
void CPUPinnedAllocator::Free(Allocation *allocation) {
  PADDLE_ENFORCE_NOT_NULL(dynamic_cast<CPUPinnedAllocation *>(allocation));
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE(cudaFreeHost(allocation->ptr()));
#endif
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE(hipHostFree(allocation->ptr()));
#endif
  delete allocation;
}
Allocation *CPUPinnedAllocator::AllocateImpl(size_t size,
                                             Allocator::Attr attr) {
  // PADDLE_ENFORCE_EQ(
  //    attr, kCrossDevice,
  //    "CPUPinnedAllocator should be used for Cross-Device Communication");

  void *ptr;
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE(cudaMallocHost(&ptr, size));
#endif
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE(hipHostMalloc(&ptr, size));
#endif
  return new CPUPinnedAllocation(ptr, size);
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
