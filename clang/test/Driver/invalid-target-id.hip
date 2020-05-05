// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908xnack \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOPLUS %s

// NOPLUS: error: Unsupported CUDA gpu architecture

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:sramecc+:xnack+ \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ORDER %s

// ORDER: error: Invalid target id: gfx908:sramecc+:xnack+

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:unknown+ \
// RUN:   --offload-arch=gfx908+sramecc+unknown \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNK %s

// UNK: error: Invalid target id: gfx908:unknown+

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:sramecc+:unknown+ \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MIXED %s

// MIXED: error: Invalid target id: gfx908:sramecc+:unknown+

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900:xnack+ \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=UNSUP %s

// UNSUP: error: Invalid target id: gfx900:xnack+

/ RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900:xnack \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOSIGN %s

// NOSIGN: error: Invalid target id: gfx900:xnack

/ RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx900+xnack \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOCOLON %s

// NOCOLON: error: Unsupported CUDA gpu architecture
