; This file is used with module-flags-target-id-dst-*.ll
; RUN: true

; Invalid target id: feature must ends with +/-.
!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack" }

!llvm.module.flags = !{ !0 }
