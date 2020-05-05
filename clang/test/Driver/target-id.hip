// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx908 \
// RUN:   --offload-arch=gfx908:xnack+:sramecc+ \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx908" "-fcuda-is-device"

// CHECK: [[OPT:".*opt"]] {{.*}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx908"
// CHECK-SAME: "-o" [[OPT_906_BC:".*-gfx908-optimized.*bc"]]

// CHECK: [[LLC: ".*llc"]] [[OPT_906_BC]]
// CHECK-SAME: "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-mcpu=gfx908"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx908"
// CHECK-SAME: {{.*}} "-target-feature" "+xnack"
// CHECK-SAME: {{.*}} "-target-feature" "+sram-ecc"

// CHECK: [[OPT]] {{.*"}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx908"
// CHECK-SAME: "-o" [[OPT_906XE_BC:".*-gfx908:xnack\+:sramecc\+.*bc"]]

// CHECK: [[LLC]] [[OPT_906XE_BC]]
// CHECK-SAME: "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-mcpu=gfx908"

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-target-cpu" "gfx908"
// CHECK-SAME: {{.*}} "-target-feature" "+xnack"
// CHECK-SAME: {{.*}} "-target-feature" "-sram-ecc"

// CHECK: [[OPT]] {{.*}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx908"
// CHECK-SAME: "-o" [[OPT_906XN_BC:".*-gfx908:xnack\+:sramecc\-.*bc"]]

// CHECK: [[LLC]] [[OPT_906XN_BC]]
// CHECK-SAME: "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: {{.*}} "-mcpu=gfx908"


// CHECK: {{".*clang-offload-bundler"}}
// CHECK-SAME: "-targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx908,hip-amdgcn-amd-amdhsa-gfx908:xnack+:sramecc+,hip-amdgcn-amd-amdhsa-gfx908:xnack+:sramecc-"
