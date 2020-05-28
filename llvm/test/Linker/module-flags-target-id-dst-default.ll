; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefixes=NOCHANGE,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefixes=NOCHANGE,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-on-sramecc-off.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefixes=BOTH,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefixes=XNACK,COMMON %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=INVALID %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-none.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT2 %s

; Test target id module flags.

; NOCHANGE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908"}
; BOTH: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-"}
; XNACK: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack-"}
; COMMON: !llvm.module.flags = !{!0}

; INVALID: invalid module flag 'target-id': incorrect format ('amdgcn-amd-amdhsa--gfx908:xnack'
; CONFLICT: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdhsa--gfx908' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx90a:sramecc-' from '{{.*}}'
; CONFLICT2: cannot link '{{.*}}' which has target-id with '{{.*}}' which does not have target-id.

!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908" }

!llvm.module.flags = !{ !0 }
