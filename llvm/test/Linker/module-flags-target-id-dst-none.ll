; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-on-sramecc-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=CONFLICT %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-none.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NONE %s

; Test target id module flags.

; NONE: !llvm.module.flags = !{!0}
; NONE: !0 = !{i32 1, !"foo", i32 37}

; CONFLICT: cannot link '{{.*}}' which has target-id with '{{.*}}' which does not have target-id.

!0 = !{ i32 1, !"foo", i32 37 }

!llvm.module.flags = !{ !0 }
