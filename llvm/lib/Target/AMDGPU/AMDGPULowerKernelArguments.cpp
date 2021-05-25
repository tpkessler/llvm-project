//===-- AMDGPULowerKernelArguments.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass replaces accesses to kernel arguments with loads from
/// offsets from the kernarg base pointer.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-lower-kernel-arguments"

using namespace llvm;

namespace {

class AMDGPULowerKernelArguments : public FunctionPass{
  const GCNSubtarget *ST;

public:
  static char ID;

  AMDGPULowerKernelArguments() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.setPreservesAll();
  }

protected:
  // Check whether we should skip lowering that given argument for certain
  // workarounds.
  bool shouldSkipLowering(const Argument &Arg) const {
    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    // Skip if it's not a pointer.
    if (PT) {
      // FIXME: Hack. We rely on AssertZext to be able to fold DS addressing
      // modes on SI to know the high bits are 0 so pointer adds don't wrap. We
      // can't represent this with range metadata because it's only allowed for
      // integer types.
      if ((PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
           PT->getAddressSpace() == AMDGPUAS::REGION_ADDRESS) &&
          !ST->hasUsableDSOffset())
        return true;
    }
    return false;
  }

  bool addAliasScopeMetadata(Function &F, AAResults *AAR,
                             DominatorTree *DT) const;
};

} // end anonymous namespace

// skip allocas
static BasicBlock::iterator getInsertPt(BasicBlock &BB) {
  BasicBlock::iterator InsPt = BB.getFirstInsertionPt();
  for (BasicBlock::iterator E = BB.end(); InsPt != E; ++InsPt) {
    AllocaInst *AI = dyn_cast<AllocaInst>(&*InsPt);

    // If this is a dynamic alloca, the value may depend on the loaded kernargs,
    // so loads will need to be inserted before it.
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  return InsPt;
}

bool AMDGPULowerKernelArguments::addAliasScopeMetadata(
    Function &F, AAResults *AAR, DominatorTree *DT) const {
  // Collect 'noalias' arguments.
  SmallVector<const Argument *, 4> NoAliasArgs;
  for (const Argument &Arg : F.args()) {
    if (Arg.hasByRefAttr() || !Arg.hasNoAliasAttr())
      continue;
    // Skip argument not being lowered.
    if (shouldSkipLowering(Arg))
      continue;
    assert(isa<PointerType>(Arg.getType()));
    NoAliasArgs.push_back(&Arg);
  }

  if (NoAliasArgs.empty())
    return false;

  // 'noalias' indicates that pointer values based on the argument do not alias
  // pointer values which are not based on it. So we add a new "scope" for each
  // noalias function argument. Accesses using pointers based on that argument
  // become part of that alias scope, accesses using pointers not based on that
  // argument are tagged as noalias with that scope.

  DenseMap<const Argument *, MDNode *> NewScopes;
  MDBuilder MDB(F.getContext());

  // Create a new scope domain for this function.
  MDNode *NewDomain =
    MDB.createAnonymousAliasScopeDomain(F.getName());
  for (unsigned i = 0, e = NoAliasArgs.size(); i != e; ++i) {
    const Argument *A = NoAliasArgs[i];

    std::string Name = std::string(F.getName());
    if (A->hasName()) {
      Name += ": %";
      Name += A->getName();
    } else {
      Name += ": argument ";
      Name += utostr(i);
    }

    // Note: We always create a new anonymous root here. This is true regardless
    // of the linkage of the callee because the aliasing "scope" is not just a
    // property of the callee, but also all control dependencies in the caller.
    MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, Name);
    NewScopes.insert(std::make_pair(A, NewScope));
  }

  // Iterate over all new instructions in the map; for all memory-access
  // instructions, add the alias scope metadata.
  for (BasicBlock &BB : F) {
    for (Instruction &II : BB) {
      Instruction *I = &II;

      bool IsArgMemOnlyCall = false, IsFuncCall = false;
      SmallVector<const Value *, 2> PtrArgs;

      if (const LoadInst *LI = dyn_cast<LoadInst>(I))
        PtrArgs.push_back(LI->getPointerOperand());
      else if (const StoreInst *SI = dyn_cast<StoreInst>(I))
        PtrArgs.push_back(SI->getPointerOperand());
      else if (const VAArgInst *VAAI = dyn_cast<VAArgInst>(I))
        PtrArgs.push_back(VAAI->getPointerOperand());
      else if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I))
        PtrArgs.push_back(CXI->getPointerOperand());
      else if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I))
        PtrArgs.push_back(RMWI->getPointerOperand());
      else if (const auto *Call = dyn_cast<CallBase>(I)) {
        // If we know that the call does not access memory, then we'll still
        // know that about the inlined clone of this call site, and we don't
        // need to add metadata.
        if (Call->doesNotAccessMemory())
          continue;

        IsFuncCall = true;
        if (AAR) {
          FunctionModRefBehavior MRB = AAR->getModRefBehavior(Call);
          if (AAResults::onlyAccessesArgPointees(MRB))
            IsArgMemOnlyCall = true;
        }

        for (Value *Arg : Call->args()) {
          // We need to check the underlying objects of all arguments, not just
          // the pointer arguments, because we might be passing pointers as
          // integers, etc.
          // However, if we know that the call only accesses pointer arguments,
          // then we only need to check the pointer arguments.
          if (IsArgMemOnlyCall && !Arg->getType()->isPointerTy())
            continue;

          PtrArgs.push_back(Arg);
        }
      }

      // If we found no pointers, then this instruction is not suitable for
      // pairing with an instruction to receive aliasing metadata.
      // However, if this is a call, this we might just alias with none of the
      // noalias arguments.
      if (PtrArgs.empty() && !IsFuncCall)
        continue;

      // It is possible that there is only one underlying object, but you
      // need to go through several PHIs to see it, and thus could be
      // repeated in the Objects list.
      SmallPtrSet<const Value *, 4> ObjSet;
      SmallVector<Metadata *, 4> Scopes, NoAliases;

      SmallSetVector<const Argument *, 4> NAPtrArgs;
      for (const Value *V : PtrArgs) {
        SmallVector<const Value *, 4> Objects;
        getUnderlyingObjects(V, Objects, /* LI = */ nullptr);

        for (const Value *O : Objects)
          ObjSet.insert(O);
      }

      // Figure out if we're derived from anything that is not a noalias
      // argument.
      bool CanDeriveViaCapture = false, UsesAliasingPtr = false;
      for (const Value *V : ObjSet) {
        // Is this value a constant that cannot be derived from any pointer
        // value (we need to exclude constant expressions, for example, that
        // are formed from arithmetic on global symbols).
        bool IsNonPtrConst = isa<ConstantInt>(V) || isa<ConstantFP>(V) ||
                             isa<ConstantPointerNull>(V) ||
                             isa<ConstantDataVector>(V) || isa<UndefValue>(V);
        if (IsNonPtrConst)
          continue;

        // If this is anything other than a noalias argument, then we cannot
        // completely describe the aliasing properties using alias.scope
        // metadata (and, thus, won't add any).
        if (const Argument *A = dyn_cast<Argument>(V)) {
          if (!A->hasNoAliasAttr())
            UsesAliasingPtr = true;
        } else {
          UsesAliasingPtr = true;
        }

        // If this is not some identified function-local object (which cannot
        // directly alias a noalias argument), or some other argument (which,
        // by definition, also cannot alias a noalias argument), then we could
        // alias a noalias argument that has been captured).
        if (!isa<Argument>(V) &&
            !isIdentifiedFunctionLocal(const_cast<Value *>(V)))
          CanDeriveViaCapture = true;
      }

      // A function call can always get captured noalias pointers (via other
      // parameters, globals, etc.).
      if (IsFuncCall && !IsArgMemOnlyCall)
        CanDeriveViaCapture = true;

      // First, we want to figure out all of the sets with which we definitely
      // don't alias. Iterate over all noalias set, and add those for which:
      //   1. The noalias argument is not in the set of objects from which we
      //      definitely derive.
      //   2. The noalias argument has not yet been captured.
      // An arbitrary function that might load pointers could see captured
      // noalias arguments via other noalias arguments or globals, and so we
      // must always check for prior capture.
      for (const Argument *A : NoAliasArgs) {
        if (!ObjSet.count(A) &&
            (!CanDeriveViaCapture ||
             // It might be tempting to skip the
             // PointerMayBeCapturedBefore check if
             // A->hasNoCaptureAttr() is true, but this is
             // incorrect because nocapture only guarantees
             // that no copies outlive the function, not
             // that the value cannot be locally captured.
             !PointerMayBeCapturedBefore(A,
                                         /* ReturnCaptures */ false,
                                         /* StoreCaptures */ false, I, DT)))
          NoAliases.push_back(NewScopes[A]);
      }

      if (!NoAliases.empty())
        I->setMetadata(
            LLVMContext::MD_noalias,
            MDNode::concatenate(I->getMetadata(LLVMContext::MD_noalias),
                                MDNode::get(F.getContext(), NoAliases)));

      // Next, we want to figure out all of the sets to which we might belong.
      // We might belong to a set if the noalias argument is in the set of
      // underlying objects. If there is some non-noalias argument in our list
      // of underlying objects, then we cannot add a scope because the fact
      // that some access does not alias with any set of our noalias arguments
      // cannot itself guarantee that it does not alias with this access
      // (because there is some pointer of unknown origin involved and the
      // other access might also depend on this pointer). We also cannot add
      // scopes to arbitrary functions unless we know they don't access any
      // non-parameter pointer-values.
      bool CanAddScopes = !UsesAliasingPtr;
      if (CanAddScopes && IsFuncCall)
        CanAddScopes = IsArgMemOnlyCall;

      if (CanAddScopes)
        for (const Argument *A : NoAliasArgs) {
          if (ObjSet.count(A))
            Scopes.push_back(NewScopes[A]);
        }

      if (!Scopes.empty())
        I->setMetadata(
            LLVMContext::MD_alias_scope,
            MDNode::concatenate(I->getMetadata(LLVMContext::MD_alias_scope),
                                MDNode::get(F.getContext(), Scopes)));
    }
  }

  return true;
}

bool AMDGPULowerKernelArguments::runOnFunction(Function &F) {
  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty())
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  ST = &TM.getSubtarget<GCNSubtarget>(F);
  LLVMContext &Ctx = F.getParent()->getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  BasicBlock &EntryBlock = *F.begin();
  IRBuilder<> Builder(&*getInsertPt(EntryBlock));

  const Align KernArgBaseAlign(16); // FIXME: Increase if necessary
  const uint64_t BaseOffset = ST->getExplicitKernelArgOffset(F);

  Align MaxAlign;
  // FIXME: Alignment is broken broken with explicit arg offset.;
  const uint64_t TotalKernArgSize = ST->getKernArgSegmentSize(F, MaxAlign);
  if (TotalKernArgSize == 0)
    return false;

  CallInst *KernArgSegment =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_kernarg_segment_ptr, {}, {},
                              nullptr, F.getName() + ".kernarg.segment");

  KernArgSegment->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
  KernArgSegment->addAttribute(AttributeList::ReturnIndex,
    Attribute::getWithDereferenceableBytes(Ctx, TotalKernArgSize));

  unsigned AS = KernArgSegment->getType()->getPointerAddressSpace();
  uint64_t ExplicitArgOffset = 0;

  addAliasScopeMetadata(F, AA, DT);

  for (Argument &Arg : F.args()) {
    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    MaybeAlign ABITypeAlign = IsByRef ? Arg.getParamAlign() : None;
    if (!ABITypeAlign)
      ABITypeAlign = DL.getABITypeAlign(ArgTy);

    uint64_t Size = DL.getTypeSizeInBits(ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);

    uint64_t EltOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + BaseOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + AllocSize;

    if (Arg.use_empty())
      continue;

    // If this is byval, the loads are already explicit in the function. We
    // just need to rewrite the pointer values.
    if (IsByRef) {
      Value *ArgOffsetPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".byval.kernarg.offset");

      Value *CastOffsetPtr = Builder.CreatePointerBitCastOrAddrSpaceCast(
          ArgOffsetPtr, Arg.getType());
      Arg.replaceAllUsesWith(CastOffsetPtr);
      continue;
    }

    if (shouldSkipLowering(Arg))
      continue;

    auto *VT = dyn_cast<FixedVectorType>(ArgTy);
    bool IsV3 = VT && VT->getNumElements() == 3;
    bool DoShiftOpt = Size < 32 && !ArgTy->isAggregateType();

    VectorType *V4Ty = nullptr;

    int64_t AlignDownOffset = alignDown(EltOffset, 4);
    int64_t OffsetDiff = EltOffset - AlignDownOffset;
    Align AdjustedAlign = commonAlignment(
        KernArgBaseAlign, DoShiftOpt ? AlignDownOffset : EltOffset);

    Value *ArgPtr;
    Type *AdjustedArgTy;
    if (DoShiftOpt) { // FIXME: Handle aggregate types
      // Since we don't have sub-dword scalar loads, avoid doing an extload by
      // loading earlier than the argument address, and extracting the relevant
      // bits.
      //
      // Additionally widen any sub-dword load to i32 even if suitably aligned,
      // so that CSE between different argument loads works easily.
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, AlignDownOffset,
          Arg.getName() + ".kernarg.offset.align.down");
      AdjustedArgTy = Builder.getInt32Ty();
    } else {
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".kernarg.offset");
      AdjustedArgTy = ArgTy;
    }

    if (IsV3 && Size >= 32) {
      V4Ty = FixedVectorType::get(VT->getElementType(), 4);
      // Use the hack that clang uses to avoid SelectionDAG ruining v3 loads
      AdjustedArgTy = V4Ty;
    }

    ArgPtr = Builder.CreateBitCast(ArgPtr, AdjustedArgTy->getPointerTo(AS),
                                   ArgPtr->getName() + ".cast");
    LoadInst *Load =
        Builder.CreateAlignedLoad(AdjustedArgTy, ArgPtr, AdjustedAlign);
    Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));

    MDBuilder MDB(Ctx);

    if (isa<PointerType>(ArgTy)) {
      if (Arg.hasNonNullAttr())
        Load->setMetadata(LLVMContext::MD_nonnull, MDNode::get(Ctx, {}));

      uint64_t DerefBytes = Arg.getDereferenceableBytes();
      if (DerefBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable,
          MDNode::get(Ctx,
                      MDB.createConstant(
                        ConstantInt::get(Builder.getInt64Ty(), DerefBytes))));
      }

      uint64_t DerefOrNullBytes = Arg.getDereferenceableOrNullBytes();
      if (DerefOrNullBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable_or_null,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          DerefOrNullBytes))));
      }

      unsigned ParamAlign = Arg.getParamAlignment();
      if (ParamAlign != 0) {
        Load->setMetadata(
          LLVMContext::MD_align,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          ParamAlign))));
      }
    }

    if (DoShiftOpt) {
      Value *ExtractBits = OffsetDiff == 0 ?
        Load : Builder.CreateLShr(Load, OffsetDiff * 8);

      IntegerType *ArgIntTy = Builder.getIntNTy(Size);
      Value *Trunc = Builder.CreateTrunc(ExtractBits, ArgIntTy);
      Value *NewVal = Builder.CreateBitCast(Trunc, ArgTy,
                                            Arg.getName() + ".load");
      Arg.replaceAllUsesWith(NewVal);
    } else if (IsV3) {
      Value *Shuf = Builder.CreateShuffleVector(Load, ArrayRef<int>{0, 1, 2},
                                                Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Shuf);
    } else {
      Load->setName(Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Load);
    }
  }

  KernArgSegment->addAttribute(
      AttributeList::ReturnIndex,
      Attribute::getWithAlignment(Ctx, std::max(KernArgBaseAlign, MaxAlign)));

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPULowerKernelArguments, DEBUG_TYPE,
                      "AMDGPU Lower Kernel Arguments", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPULowerKernelArguments, DEBUG_TYPE, "AMDGPU Lower Kernel Arguments",
                    false, false)

char AMDGPULowerKernelArguments::ID = 0;

FunctionPass *llvm::createAMDGPULowerKernelArgumentsPass() {
  return new AMDGPULowerKernelArguments();
}
