#include <iostream>

#include "cinnrt/host_context/kernel_registry.h"

void RegisterKernels(cinnrt::host_context::KernelRegistry *registry);
void RegisterBasicKernels(cinnrt::host_context::KernelRegistry *registry);
void RegisterPaddleKernels(cinnrt::host_context::KernelRegistry *registry);
