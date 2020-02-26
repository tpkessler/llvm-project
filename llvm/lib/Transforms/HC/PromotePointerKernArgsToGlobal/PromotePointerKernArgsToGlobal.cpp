//===--  PromotePointerKernargsToGlobal.cpp - Promote Pointers To Global --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares and defines a pass which uses the double-cast trick (
// generic-to-global and global-to-generic) for the formal arguments of pointer
// type of a kernel (i.e. pfe trampoline or HIP __global__ function). This
// transformation is valid due to the invariants established by both HC and HIP
// in accordance with an address passed to a kernel can only reside in the
// global address space. It is preferable to execute SelectAcceleratorCode
// before, as this reduces the workload by pruning functions that are not
// reachable by an accelerator. It is mandatory to run InferAddressSpaces after,
// otherwise no benefit shall be obtained (the spurious casts do get removed).
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

#include <algorithm>

using namespace llvm;
using namespace std;

namespace {
class PromotePointerKernArgsToGlobal : public FunctionPass {
    // TODO: query the address space robustly.
    static constexpr unsigned int GenericAddrSpace{0u};
    static constexpr unsigned int GlobalAddrSpace{1u};
public:
    static char ID;
    PromotePointerKernArgsToGlobal() : FunctionPass{ID} {}

    bool runOnFunction(Function &F) override
    {
        if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
            return false;

        SmallVector<User *, 8> Promotable;
        for (auto &&Arg : F.args()) {
            for (auto &&U : Arg.users()) {
                if (!U->getType()->isPointerTy())
                    continue;
                if (U->getType()->getPointerAddressSpace() != GenericAddrSpace)
                    continue;

                Promotable.push_back(U);
            }
        }

        if (Promotable.empty())
            return false;

        static IRBuilder<> Builder{F.getContext()};
        for (auto &&PU : Promotable) {
            if (!isa<Instruction>(PU))
                continue;

            Builder.SetInsertPoint(
                cast<Instruction>(PU)->getNextNonDebugInstruction());

            Value *TmpAlloca = Builder.CreateAlloca(PU->getType());
            Value *Tmp = Builder.CreateLoad(TmpAlloca, "Tmp");

            PU->replaceAllUsesWith(Tmp);

            Value *FToG = Builder.CreateAddrSpaceCast(
                PU,
                cast<PointerType>(PU->getType())
                    ->getElementType()->getPointerTo(GlobalAddrSpace));
            Value *GToF = Builder.CreateAddrSpaceCast(FToG, PU->getType());

            Tmp->replaceAllUsesWith(GToF);
        }

        return true;
    }
};
char PromotePointerKernArgsToGlobal::ID = 0;

static RegisterPass<PromotePointerKernArgsToGlobal> X{
    "promote-pointer-kernargs-to-global",
    "Promotes kernel formals of pointer type to point to the global address "
    "space, since the actuals can only represent a global address.",
    false,
    false};
}