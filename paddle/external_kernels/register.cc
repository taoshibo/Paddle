#include "paddle/external_kernels/register.h"

void RegisterKernels(cinnrt::host_context::KernelRegistry *registry) {
  RegisterBasicKernels(registry);
  RegisterPaddleKernels(registry);
}
