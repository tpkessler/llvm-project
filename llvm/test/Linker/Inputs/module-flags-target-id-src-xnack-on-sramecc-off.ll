; This file is used with module-flags-target-id-dst-*.ll
; RUN: true

!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-" }

!llvm.module.flags = !{ !0 }
