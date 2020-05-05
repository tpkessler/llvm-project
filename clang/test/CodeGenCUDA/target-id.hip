// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -target-cpu gfx908 -target-feature +xnack \
// RUN:   -target-feature -sramecc \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: !{i32 8, !"target-id", !"amdgcn-amd-amdhsa-gfx908:xnack+:sramecc-"}
__global__ void foo() {}
