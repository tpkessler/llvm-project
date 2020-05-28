; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefix=NOCHANGE %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefixes=NOCHANGE %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-on-sramecc-off.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefix=NOCHANGE %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  | sort | FileCheck -check-prefix=NOCHANGE %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NOXNACK %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=INVALID %s

; Test target id module flags.

; NOCHANGE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-"}
; NOCHANGE: !llvm.module.flags = !{!0}

; NOXNACK: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908:xnack-' from '{{.*}}'

; INVALID: invalid module flag 'target-id': incorrect format ('amdgcn-amd-amdhsa--gfx908:xnack'

!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-" }

!llvm.module.flags = !{ !0 }
